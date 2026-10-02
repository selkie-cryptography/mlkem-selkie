// Keeps one open issue per check and updates it from CI results.
//
// Load from actions/github-script in a job that holds `issues: write` and
// runs no cargo. Each producer leg writes a directory named after the leg:
//   status       `pass` or `fail`
//   version.txt  toolchain or tool version
//   output.txt   captured output
// Producers upload the parent of that directory; the reporter downloads all
// of them with `merge-multiple: true`, so leg names must be unique.
// `legsFromNeeds` builds legs from job results alone, with no artifacts.
//
// On failure: opens an issue, or comments on the open one when the output
// hash changed. On pass: closes the open issue. A missing result leaves the
// issue alone and fails the step. Runs off the default branch write nothing.
//
// The issue body and each update comment start with a marker holding the key
// and output hash, so the reporter needs no other state. Only issues and
// comments by `author` count, so a pasted marker can't redirect it.

'use strict';

const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');

const DEFAULT_AUTHOR = 'github-actions[bot]';
// GitHub caps issue bodies at 65536 characters.
const EXCERPT_BUDGET = 40000;
const KEY = /^[A-Za-z0-9._/-]+$/;
const HASH = /^[0-9a-f]{64}$/;
const BODY_MARKER = /^<!-- track-issue key=(\S+) hash=([0-9a-f]{64}) -->/;
const COMMENT_MARKER = /^<!-- track-issue hash=([0-9a-f]{64}) -->/;
const NEWLINE = /\r\n|\r|\n/;
// cargo progress lines change run to run (timings, build order, hashes).
const NOISE = /^\s*(Adding|Blocking|Checking|Compiling|Doc-tests|Documenting|Downloaded|Downloading|Finished|Fresh|Installed|Installing|Locking|Running|Updating)\s/;
const ANSI = /\x1b\[[0-9;]*[A-Za-z]/g;
// ICE reports name a file with a timestamp and PID, the rustc build, and
// compiler source lines; all change as the floating channel moves.
const ICE_FILE = /rustc-ice-[\w.-]+\.txt/g;
const ICE_RUSTC = /^note: rustc .* running on .*$/gm;
const COMPILER_LINE = /(compiler\/[\w./-]+\.rs):\d+(:\d+)?/g;
const CONTROL = /[\x00-\x1f\x7f]/g;
// Reads stop here, so a runaway or hostile output can't exhaust the reporter.
const MAX_OUTPUT = 1 << 20;

/** Output lines without ANSI codes, cargo progress, or repeated blank lines. */
function normalize(text) {
  const lines = text
    .replace(ANSI, '')
    .replace(ICE_FILE, 'rustc-ice-*.txt')
    .replace(ICE_RUSTC, 'note: rustc *')
    .replace(COMPILER_LINE, '$1')
    .split(NEWLINE)
    .map((line) => line.trimEnd())
    .filter((line) => !NOISE.test(line))
    .filter((line, i, all) => line || (i > 0 && all[i - 1]));
  while (lines.length && !lines[0]) lines.shift();
  while (lines.length && !lines[lines.length - 1]) lines.pop();

  return lines;
}

/**
 * SHA-256 over each leg's name, status, and sorted unique output lines.
 * Sorting ignores the interleaving of parallel builds.
 */
function fingerprint(legs) {
  const hash = crypto.createHash('sha256');
  const sorted = [...legs].sort((a, b) => a.name.localeCompare(b.name));
  for (const leg of sorted) {
    const lines = [...new Set(normalize(leg.output))].sort();
    hash.update(`${leg.name}\0${leg.status}\0${lines.join('\n')}\0`);
  }
  return hash.digest('hex');
}

/** Reads a producer's result directory; status is `missing` without one. */
function readResult(dir) {
  const read = (name) => {
    let fd;
    try {
      fd = fs.openSync(path.join(dir, name), 'r');
      const buf = Buffer.alloc(MAX_OUTPUT + 1);
      const n = fs.readSync(fd, buf, 0, buf.length, 0);
      const text = buf.toString('utf8', 0, Math.min(n, MAX_OUTPUT));
      return n > MAX_OUTPUT ? `${text}\n... truncated at ${MAX_OUTPUT} bytes` : text;
    } catch {
      return '';
    } finally {
      if (fd !== undefined) fs.closeSync(fd);
    }
  };
  const status = read('status').trim();

  return {
    status: status === 'pass' || status === 'fail' ? status : 'missing',
    version: read('version.txt').trim(),
    output: read('output.txt'),
  };
}

/**
 * Legs from a `needs` context (`toJSON(needs)`): a successful job passes, a
 * failed one fails, and a cancelled or skipped one is missing.
 */
function legsFromNeeds(needs) {
  const status = { success: 'pass', failure: 'fail' };

  return Object.entries(needs).map(([name, job]) => ({
    name,
    status: status[job.result] || 'missing',
    version: '',
    output: '',
  }));
}

/** Backticks for a fence longer than any backtick run in `text`. */
function fenceTicks(text) {
  let longest = 0;
  let run = 0;
  for (const ch of text) {
    run = ch === '`' ? run + 1 : 0;
    if (run > longest) longest = run;
  }

  return '`'.repeat(Math.max(3, longest + 1));
}

/** `text` in a fenced block it can't close. */
function fence(text) {
  const ticks = fenceTicks(text);

  return `${ticks}text\n${text}\n${ticks}`;
}

/** The first lines of normalized output that fit in `budget` characters. */
function excerpt(output, budget) {
  const lines = normalize(output);
  const kept = [];
  let used = 0;
  for (const line of lines) {
    if (used + line.length + 1 > budget) break;
    kept.push(line);
    used += line.length + 1;
  }
  const dropped = lines.length - kept.length;
  if (dropped > 0) kept.push(`... ${dropped} more lines; full output is in the run's artifacts.`);

  return kept.join('\n') || '(no output)';
}

/** A fenced excerpt whose fence counts against `budget`. */
function quote(output, budget) {
  const room = budget - 2 * fenceTicks(normalize(output).join('\n')).length - 16;

  return fence(room > 0 ? excerpt(output, room) : "(output too large to quote; see the run's artifacts)");
}

/** One-line inline code, safe in a table cell for any version string. */
function code(text) {
  const line = (text || 'unknown')
    .split(NEWLINE)[0]
    .slice(0, 200)
    .replace(CONTROL, ' ')
    .replace(/`/g, "'")
    .replace(/\|/g, '\\|');

  return `\`${line}\``;
}

/**
 * Markdown for the failing state: intro, per-leg status, output, run link.
 * Drops the version column and output blocks when the legs have none.
 */
function render({ intro, legs, runUrl }) {
  const failing = legs.filter((leg) => leg.status === 'fail' && leg.output);
  const budget = Math.floor(EXCERPT_BUDGET / Math.max(1, failing.length));
  const versioned = legs.some((leg) => leg.version);
  const parts = [intro, ''];
  if (versioned) {
    parts.push('| Leg | Status | Version |', '| --- | --- | --- |');
    for (const leg of legs) parts.push(`| ${leg.name} | ${leg.status} | ${code(leg.version)} |`);
  } else {
    parts.push('| Leg | Status |', '| --- | --- |');
    for (const leg of legs) parts.push(`| ${leg.name} | ${leg.status} |`);
  }
  for (const leg of failing) {
    parts.push('', `<details><summary>${leg.name} output</summary>`, '', quote(leg.output, budget), '', '</details>');
  }
  parts.push('', `Run: ${runUrl}`);

  return parts.join('\n');
}

/** Open issues with `label` whose marker names `key`, split by author. */
async function findOpenIssues(github, owner, repo, { key, label, author }) {
  const issues = await github.paginate(github.rest.issues.listForRepo, {
    owner,
    repo,
    state: 'open',
    labels: label,
    per_page: 100,
  });
  const matching = issues
    .filter((issue) => !issue.pull_request)
    .filter((issue) => {
      const match = BODY_MARKER.exec(issue.body || '');
      return match && match[1] === key;
    })
    .sort((a, b) => a.number - b.number);
  const byAuthor = (issue) => issue.user && issue.user.login === author;

  return { mine: matching.find(byAuthor), others: matching.filter((issue) => !byAuthor(issue)) };
}

/** The newest hash: the last marked comment by `author`, else the body's. */
async function lastHash(github, owner, repo, issue, author) {
  const comments = await github.paginate(github.rest.issues.listComments, {
    owner,
    repo,
    issue_number: issue.number,
    per_page: 100,
  });
  for (const comment of comments.reverse()) {
    if (!comment.user || comment.user.login !== author) continue;
    const match = COMMENT_MARKER.exec(comment.body || '');
    if (match) return match[1];
  }

  return BODY_MARKER.exec(issue.body || '')[2];
}

async function ensureLabel(github, owner, repo, name) {
  try {
    await github.rest.issues.getLabel({ owner, repo, name });
  } catch (error) {
    if (error.status !== 404) throw error;
    try {
      await github.rest.issues.createLabel({
        owner,
        repo,
        name,
        color: 'd93f0b',
        description: 'Opened and closed by CI (.github/scripts/track-issue.js)',
      });
    } catch (createError) {
      // 422: created concurrently.
      if (createError.status !== 422) throw createError;
    }
  }
}

/**
 * Syncs one tracked issue with a check's legs.
 *
 * Returns `missing`, `conflict`, `pass`, `closed`, `opened`, `commented`, or
 * `unchanged`.
 */
async function trackIssue({ github, context, core }, opts) {
  const { key, title, label, intro, legs, runUrl } = opts;
  const author = opts.author || DEFAULT_AUTHOR;
  if (!KEY.test(key)) throw new Error(`bad key ${JSON.stringify(key)}`);
  if (!legs.length) throw new Error(`${key}: no legs`);
  const { owner, repo } = context.repo;

  const missing = legs.filter((leg) => leg.status === 'missing').map((leg) => leg.name);
  if (missing.length) {
    core.setFailed(`${key}: no result from ${missing.join(', ')}; issue left unchanged`);
    return 'missing';
  }

  const { mine: issue, others } = await findOpenIssues(github, owner, repo, { key, label, author });
  const failed = legs.some((leg) => leg.status === 'fail');

  if (!failed) {
    if (!issue) return 'pass';
    const versions = [...new Set(legs.filter((leg) => leg.version).map((leg) => code(leg.version)))];
    const on = versions.length ? ` on ${versions.join(', ')}` : '';
    await github.rest.issues.createComment({
      owner,
      repo,
      issue_number: issue.number,
      body: `Passes${on}. Closing.\n\nRun: ${runUrl}`,
    });
    await github.rest.issues.update({
      owner,
      repo,
      issue_number: issue.number,
      state: 'closed',
      state_reason: 'completed',
    });
    core.info(`${key}: closed #${issue.number}`);
    return 'closed';
  }

  const hash = fingerprint(legs);
  if (!HASH.test(hash)) throw new Error('bad hash');
  const text = render({ intro, legs, runUrl });

  if (!issue) {
    // A labeled issue for this key by another author means the token changed
    // or someone copied the marker; fail instead of opening a duplicate.
    if (others.length) {
      const numbers = others.map((other) => `#${other.number}`).join(', ');
      core.setFailed(`${key}: open issue ${numbers} is not by ${author}; not opening another`);
      return 'conflict';
    }
    await ensureLabel(github, owner, repo, label);
    const { data } = await github.rest.issues.create({
      owner,
      repo,
      title,
      labels: [label],
      body: `<!-- track-issue key=${key} hash=${hash} -->\n${text}`,
    });
    core.info(`${key}: opened #${data.number}`);
    return 'opened';
  }

  if ((await lastHash(github, owner, repo, issue, author)) === hash) {
    core.info(`${key}: #${issue.number} unchanged`);
    return 'unchanged';
  }
  await github.rest.issues.createComment({
    owner,
    repo,
    issue_number: issue.number,
    body: `<!-- track-issue hash=${hash} -->\nOutput changed.\n\n${text}`,
  });
  core.info(`${key}: commented on #${issue.number}`);
  return 'commented';
}

/**
 * Runs `trackIssue` for each tracker; one failure doesn't skip the rest.
 * Off the default branch it only logs, so a branch run can't close or open
 * the issues that track the default branch.
 */
async function trackAll(ctx, trackers) {
  const { github, context, core } = ctx;
  const { data } = await github.rest.repos.get(context.repo);
  if (context.ref !== `refs/heads/${data.default_branch}`) {
    for (const tracker of trackers) {
      const states = tracker.legs.map((leg) => `${leg.name}=${leg.status}`).join(' ');
      core.notice(`${tracker.key}: ${states}; not on ${data.default_branch}, issues unchanged`);
    }
    return;
  }
  for (const tracker of trackers) {
    try {
      await trackIssue(ctx, tracker);
    } catch (error) {
      core.setFailed(`${tracker.key}: ${error.message}`);
    }
  }
}

module.exports = {
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
};
