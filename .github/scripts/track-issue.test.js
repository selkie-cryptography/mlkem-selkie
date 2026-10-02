// Run with `node --test .github/scripts/track-issue.test.js`.

'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const {
  trackIssue,
  trackAll,
  readResult,
  legsFromNeeds,
  normalize,
  fingerprint,
  excerpt,
  fence,
  code,
  render,
} = require('./track-issue.js');

const BOT = 'github-actions[bot]';
const RUN = 'https://github.com/o/r/actions/runs/1';

/** In-memory issues API: records writes, serves reads from `issues`. */
function fakeGithub() {
  const state = { issues: [], comments: {}, labels: new Set(), writes: [], next: 1 };
  const page = (fn) => async (params) => ({ data: fn(params) });
  const rest = {
    repos: { get: async () => ({ data: { default_branch: 'main' } }) },
    issues: {
      listForRepo: page(({ labels }) =>
        state.issues.filter((i) => i.state === 'open' && i.labels.includes(labels)),
      ),
      listComments: page(({ issue_number }) => [...(state.comments[issue_number] || [])]),
      getLabel: async ({ name }) => {
        if (!state.labels.has(name)) throw Object.assign(new Error('Not Found'), { status: 404 });
        return { data: { name } };
      },
      createLabel: async ({ name }) => {
        state.writes.push(['createLabel', name]);
        state.labels.add(name);
      },
      create: async ({ title, body, labels }) => {
        const issue = { number: state.next++, title, body, labels, state: 'open', user: { login: BOT } };
        state.issues.push(issue);
        state.writes.push(['create', issue.number]);
        return { data: issue };
      },
      createComment: async ({ issue_number, body }) => {
        (state.comments[issue_number] ||= []).push({ body, user: { login: BOT } });
        state.writes.push(['comment', issue_number]);
      },
      update: async ({ issue_number, state: s }) => {
        state.issues.find((i) => i.number === issue_number).state = s;
        state.writes.push(['update', issue_number, s]);
      },
    },
  };
  const github = { rest, paginate: async (fn, params) => (await fn(params)).data };

  return { github, state };
}

function ctx(github, ref = 'refs/heads/main') {
  const failures = [];
  const notices = [];
  const core = { info() {}, notice: (m) => notices.push(m), setFailed: (m) => failures.push(m) };

  return { ctx: { github, core, context: { ref, repo: { owner: 'o', repo: 'r' } } }, failures, notices };
}

function opts(legs) {
  return { key: 'canary/clippy', title: 'clippy canary', label: 'toolchain-canary', intro: 'Fails.', legs, runUrl: RUN };
}

const fail = (output) => ({ name: 'portable', status: 'fail', version: 'rustc 1.100.0-beta.3', output });
const pass = { name: 'portable', status: 'pass', version: 'rustc 1.100.0-beta.3', output: '' };

test('normalize drops ANSI codes, cargo progress, and repeated blanks', () => {
  const out = '\n\x1b[1mwarning\x1b[0m: x\n\n\n   Compiling foo v1.0.0\n    Finished `dev` in 3.2s\nerror: y  \r\rz\n\n';
  assert.deepEqual(normalize(out), ['warning: x', '', 'error: y', '', 'z']);
});

test('fingerprint ignores line order, progress lines, and ICE metadata', () => {
  const a = fingerprint([fail('   Checking a\nwarning: one\nwarning: two\n    Finished in 1s')]);
  const b = fingerprint([fail('warning: two\nwarning: one\n    Finished in 9s')]);
  const c = fingerprint([fail('warning: three')]);
  assert.equal(a, b);
  assert.notEqual(a, c);

  const ice = (stamp, build, line) =>
    fingerprint([
      fail(
        [
          `thread 'rustc' panicked at compiler/rustc_middle/src/ty/mod.rs:${line}:9:`,
          `note: please attach the file at \`/w/rustc-ice-${stamp}.txt\``,
          `note: rustc 1.100.0-beta.${build} (abc${build} 2026-10-0${build}) running on x86_64-unknown-linux-gnu`,
        ].join('\n'),
      ),
    ]);
  assert.equal(ice('2026-10-05T07_01_02-1234', 3, 812), ice('2026-10-12T07_03_09-5678', 4, 820));
});

test('fence outgrows backtick runs in the output', () => {
  assert.ok(fence('a ```` b').startsWith('`````text\n'));
  assert.ok(fence('plain').startsWith('```text\n'));
});

test('excerpt truncates at a line boundary', () => {
  const out = excerpt('aaaa\nbbbb\ncccc\n', 10);
  assert.equal(out, "aaaa\nbbbb\n... 1 more lines; full output is in the run's artifacts.");
});

test('code keeps one line, drops control characters, and escapes pipes', () => {
  assert.equal(code('rustc 1.0\r\r@octocat [x](https://e.example)'), '`rustc 1.0`');
  assert.equal(code('a\tb|c`d'), "`a b\\|c'd`");
  assert.equal(code(''), '`unknown`');
});

test('render stays under the issue body cap for hostile output', () => {
  const huge = '`'.repeat(13000) + '\n' + 'x'.repeat(30000);
  const legs = ['a', 'b', 'c'].map((name) => ({ name, status: 'fail', version: 'v', output: huge }));
  assert.ok(render({ intro: 'Fails.', legs, runUrl: RUN }).length < 60000);
});

test('fence scans many backtick runs without overflowing the stack', () => {
  const text = '`a'.repeat(200000) + '````';
  assert.ok(fence(text).startsWith('`````text\n'));
});

test('readResult truncates large output', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'track-issue-'));
  fs.writeFileSync(path.join(dir, 'status'), 'fail\n');
  fs.writeFileSync(path.join(dir, 'output.txt'), 'x'.repeat((1 << 20) + 10));
  const { output } = readResult(dir);
  assert.ok(output.length < (1 << 20) + 100);
  assert.match(output, /truncated at 1048576 bytes$/);
});

test('legsFromNeeds maps job results to leg states', () => {
  const needs = {
    'cargo-deny': { result: 'failure', outputs: {} },
    passed: { result: 'success', outputs: {} },
    other: { result: 'cancelled', outputs: {} },
    skipped: { result: 'skipped', outputs: {} },
  };
  assert.deepEqual(
    legsFromNeeds(needs).map((leg) => `${leg.name}=${leg.status}`),
    ['cargo-deny=fail', 'passed=pass', 'other=missing', 'skipped=missing'],
  );
});

test('render omits empty version and output columns', () => {
  const legs = legsFromNeeds({ a: { result: 'failure' }, b: { result: 'success' } });
  const text = render({ intro: 'Fails.', legs, runUrl: RUN });
  assert.match(text, /^\| Leg \| Status \|$/m);
  assert.doesNotMatch(text, /Version|<details>/);
});

test('readResult reports a missing directory as missing', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'track-issue-'));
  assert.equal(readResult(path.join(dir, 'absent')).status, 'missing');
  fs.writeFileSync(path.join(dir, 'status'), 'fail\n');
  fs.writeFileSync(path.join(dir, 'output.txt'), 'warning: x\n');
  assert.deepEqual(readResult(dir), { status: 'fail', version: '', output: 'warning: x\n' });
  fs.writeFileSync(path.join(dir, 'status'), 'maybe\n');
  assert.equal(readResult(dir).status, 'missing');
});

test('opens once, comments only on a new hash, closes on pass', async () => {
  const { github, state } = fakeGithub();
  const { ctx: c } = ctx(github);

  assert.equal(await trackIssue(c, opts([fail('warning: one')])), 'opened');
  assert.match(state.issues[0].body, /^<!-- track-issue key=canary\/clippy hash=[0-9a-f]{64} -->\n/);
  assert.deepEqual(state.issues[0].labels, ['toolchain-canary']);

  assert.equal(await trackIssue(c, opts([fail('warning: one')])), 'unchanged');
  assert.equal(await trackIssue(c, opts([fail('warning: two')])), 'commented');
  assert.equal(await trackIssue(c, opts([fail('warning: two')])), 'unchanged');

  assert.equal(await trackIssue(c, opts([pass])), 'closed');
  assert.equal(state.issues[0].state, 'closed');
  assert.equal(await trackIssue(c, opts([pass])), 'pass');

  assert.deepEqual(state.writes, [
    ['createLabel', 'toolchain-canary'],
    ['create', 1],
    ['comment', 1],
    ['comment', 1],
    ['update', 1, 'closed'],
  ]);
});

test('ignores comments by other authors', async () => {
  const { github, state } = fakeGithub();
  const { ctx: c } = ctx(github);

  assert.equal(await trackIssue(c, opts([fail('warning: one')])), 'opened');
  state.comments[1] = [
    { user: { login: 'someone' }, body: `<!-- track-issue hash=${fingerprint([fail('warning: two')])} -->` },
  ];
  assert.equal(await trackIssue(c, opts([fail('warning: two')])), 'commented');
});

test('a labeled issue by another author blocks a duplicate', async () => {
  const { github, state } = fakeGithub();
  const { ctx: c, failures } = ctx(github);
  state.issues.push({
    number: 7,
    labels: ['toolchain-canary'],
    state: 'open',
    user: { login: 'someone' },
    body: `<!-- track-issue key=canary/clippy hash=${'0'.repeat(64)} -->`,
  });

  assert.equal(await trackIssue(c, opts([fail('warning: one')])), 'conflict');
  assert.match(failures[0], /#7/);
  assert.deepEqual(state.writes, []);
});

test('a missing leg fails the step and writes nothing', async () => {
  const { github, state } = fakeGithub();
  const { ctx: c, failures } = ctx(github);
  const legs = [fail('warning: one'), { name: 'neon', status: 'missing', version: '', output: '' }];

  assert.equal(await trackIssue(c, opts(legs)), 'missing');
  assert.equal(failures.length, 1);
  assert.deepEqual(state.writes, []);
});

test('trackAll writes nothing off the default branch', async () => {
  const { github, state } = fakeGithub();
  const { ctx: c, notices } = ctx(github, 'refs/heads/bump-pins');

  await trackAll(c, [opts([fail('warning: one')])]);
  assert.deepEqual(state.writes, []);
  assert.match(notices[0], /portable=fail; not on main/);
});

test('trackAll reports each tracker on the default branch', async () => {
  const { github, state } = fakeGithub();
  const { ctx: c, failures } = ctx(github);

  await trackAll(c, [{ ...opts([pass]), key: 'bad key' }, opts([fail('warning: one')])]);
  assert.match(failures[0], /bad key/);
  assert.deepEqual(state.writes.map((w) => w[0]), ['createLabel', 'create']);
});

test('rejects keys that could break the marker', async () => {
  const { github } = fakeGithub();
  const { ctx: c } = ctx(github);
  await assert.rejects(trackIssue(c, { ...opts([pass]), key: 'a b -->' }), /bad key/);
});
