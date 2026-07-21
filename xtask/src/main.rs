//! Cargo xtask driver for the SLOTHY scheduling workflow.
//!
//! Automates the pipeline documented in `tools/slothy/README.md`:
//!
//! 1. `cargo xtask slothy run <kernel>` — invoke `slothy-cli` (from the fork
//!    checkout at `$SLOTHY_DIR`) on the kernel's input `.s`, archive the
//!    scheduled output next to it, then splice the result into the crate.
//! 2. `cargo xtask slothy gen <kernel>` — splice from the archived output
//!    without invoking SLOTHY (no Python needed; deterministic).
//! 3. `cargo xtask slothy check` — regenerate every kernel from its archived
//!    output and fail if the spliced code in the crate has drifted.
//!
//! Splicing rewrites only the `core::arch::asm!(...)` invocation inside the
//! kernel's named function (located syntactically — no marker comments in
//! the production source). The surrounding rustdoc, fn signature, and SAFETY
//! comments are never touched.
//!
//! The transcription mechanics this replaces (previously done by hand):
//! operand substitution (`x0` → `{ptr}`), loop-label conversion to an
//! `asm!`-safe numeric label (`2:` / `2b`), and deriving the clobber list
//! from the registers the schedule actually touches.

use std::{
    env, fs,
    path::{Path, PathBuf},
    process::{Command, ExitCode},
};

/// One architectural register in the kernel `.s` that maps to an `asm!`
/// operand in the generated block.
struct Operand {
    /// Register name as written in the kernel, e.g. `"x0"`.
    reg: &'static str,
    /// The `asm!` template placeholder it becomes, e.g. `"{ptr}"`.
    placeholder: &'static str,
    /// The full operand binding line emitted after the instructions.
    binding: &'static str,
}

/// A SLOTHY-scheduled kernel: where its sources live and how its registers
/// map onto the Rust function it is spliced into.
struct KernelSpec {
    /// Kernel name; also the marker name in the target Rust file.
    name: &'static str,
    /// Input kernel, repo-relative.
    input: &'static str,
    /// Archived SLOTHY output, repo-relative.
    output: &'static str,
    /// Function label opening the kernel in the `.s` files.
    fn_label: &'static str,
    /// Loop label SLOTHY software-pipelines (`-l`).
    loop_label: &'static str,
    /// Rust file containing the spliced `asm!` block, repo-relative.
    target_file: &'static str,
    /// Function in `target_file` whose `asm!` invocation is the splice
    /// target.
    target_fn: &'static str,
    /// Register → `asm!` operand mapping.
    operands: &'static [Operand],
}

/// The SLOTHY target microarchitecture model.
const MODEL: &str = "Apple_M4_everest_experimental";

/// Registers SLOTHY must not allocate: aarch64 callee-saved `v8`-`v15`, so
/// the spliced `asm!` clobbers only caller-saved state.
const RESERVED: &str = "[v8,v9,v10,v11,v12,v13,v14,v15]";

/// Every kernel under xtask management.
const KERNELS: &[KernelSpec] = &[
    KernelSpec {
        name: "ntt_stride128",
        input: "tools/slothy/stride128_input.s",
        output: "tools/slothy/stride128_output_m4.s",
        fn_label: "ntt_stride128",
        loop_label: "stride128_start",
        target_file: "src/algebraic/poly/arch/neon.rs",
        target_fn: "ntt_stride128_asm",
        operands: &[
            Operand {
                reg: "x0",
                placeholder: "{ptr}",
                binding: "ptr  = inout(reg) ptr => _,",
            },
            Operand {
                reg: "w1",
                placeholder: "{zeta:w}",
                binding: "zeta = in(reg) zeta as u32,",
            },
            Operand {
                reg: "w2",
                placeholder: "{zbar:w}",
                binding: "zbar = in(reg) zeta_bar as u32,",
            },
        ],
    },
    KernelSpec {
        name: "ntt_stride64",
        input: "tools/slothy/stride64_input.s",
        output: "tools/slothy/stride64_output_m4.s",
        fn_label: "ntt_stride64_group",
        loop_label: "stride64_start",
        target_file: "src/algebraic/poly/arch/neon.rs",
        target_fn: "ntt_stride64_group_asm",
        operands: &[
            Operand {
                reg: "x0",
                placeholder: "{ptr}",
                binding: "ptr  = inout(reg) ptr => _,",
            },
            Operand {
                reg: "w1",
                placeholder: "{zeta:w}",
                binding: "zeta = in(reg) zeta as u32,",
            },
            Operand {
                reg: "w2",
                placeholder: "{zbar:w}",
                binding: "zbar = in(reg) zeta_bar as u32,",
            },
        ],
    },
    // Test-only: thinned inverse stride-128 stage (no sum-path reduction,
    // per the lazy len-2/len-16 schedule). Splices into the test module.
    KernelSpec {
        name: "intt_stride128_thin",
        input: "tools/slothy/intt_stride128_thin_input.s",
        output: "tools/slothy/intt_stride128_thin_output_m4.s",
        fn_label: "intt_stride128",
        loop_label: "intt_stride128_start",
        target_file: "src/algebraic/poly/arch/neon/tests.rs",
        target_fn: "intt_stride128_thin_asm",
        operands: &[
            Operand {
                reg: "x0",
                placeholder: "{ptr}",
                binding: "ptr  = inout(reg) ptr => _,",
            },
            Operand {
                reg: "w1",
                placeholder: "{zeta:w}",
                binding: "zeta = in(reg) zeta as u32,",
            },
            Operand {
                reg: "w2",
                placeholder: "{zbar:w}",
                binding: "zbar = in(reg) zeta_bar as u32,",
            },
        ],
    },
];

impl KernelSpec {
    fn by_name(name: &str) -> Option<&'static KernelSpec> {
        KERNELS.iter().find(|k| k.name == name)
    }

    /// Invokes `slothy-cli` from the fork checkout on this kernel's input,
    /// writing the schedule to `self.output`.
    fn run_slothy(&self, root: &Path) -> Result<(), String> {
        let slothy_dir = slothy_dir()?;
        let python = slothy_dir.join(".venv/bin/python");
        let cli = slothy_dir.join("slothy-cli");
        for (path, hint) in [
            (
                &python,
                "create it with `python3 -m venv .venv` + `pip install -e . ortools sympy`",
            ),
            (&cli, "is $SLOTHY_DIR a SLOTHY checkout?"),
        ] {
            if !path.exists() {
                return Err(format!("{} not found ({hint})", path.display()));
            }
        }

        let status = Command::new(python)
            .arg(cli)
            .args(["Arm_AArch64", MODEL])
            .arg(root.join(self.input))
            .args(["-c", "sw_pipelining.enabled=true"])
            .args(["-c", "inputs_are_outputs"])
            .args(["-c", "sw_pipelining.allow_post=true"])
            .args(["-c", "variable_size"])
            .args(["-c", "constraints.stalls_first_attempt=8"])
            .args(["-c", &format!("reserved_regs={RESERVED}")])
            .args(["-l", self.loop_label])
            .arg("-o")
            .arg(root.join(self.output))
            .status()
            .map_err(|e| format!("failed to spawn slothy-cli: {e}"))?;
        if !status.success() {
            return Err(format!("slothy-cli failed with {status}"));
        }
        Ok(())
    }

    /// Transcribes the archived SLOTHY output into the `asm!` region text
    /// (everything between the markers, marker lines excluded).
    fn transcribe(&self, root: &Path) -> Result<String, String> {
        let output = read(&root.join(self.output))?;
        let input = read(&root.join(self.input))?;

        // Prologue length = instruction lines between the fn label and the
        // loop label in the *input*; SLOTHY passes the prologue through
        // verbatim, so this splits prologue from SLOTHY preamble on output.
        let prologue_len = region_lines(&input, self.fn_label)?
            .iter()
            .take_while(|l| l.trim() != format!("{}:", self.loop_label))
            .filter(|l| is_instruction(l))
            .count();

        let mut head = Vec::new();
        let mut body = Vec::new();
        let mut tail = Vec::new();
        let mut in_body = false;
        let mut past_backedge = false;
        let mut body_cycles: Option<u32> = None;
        let mut body_ipc: Option<String> = None;

        for line in region_lines(&output, self.fn_label)? {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            if let Some(comment) = trimmed.strip_prefix("//") {
                // The first stats block after the loop label describes the
                // software-pipelined steady-state body.
                if in_body && !past_backedge {
                    let comment = comment.trim();
                    if let Some(cy) = comment.strip_prefix("Expected cycles:") {
                        body_cycles.get_or_insert_with(|| cy.trim().parse().unwrap_or(0));
                    }
                    if let Some(ipc) = comment.strip_prefix("Expected IPC:") {
                        body_ipc.get_or_insert_with(|| ipc.trim().to_string());
                    }
                }
                continue;
            }
            if trimmed == format!("{}:", self.loop_label) {
                in_body = true;
                continue;
            }
            let instr = strip_comment(trimmed);
            if instr.is_empty() {
                continue;
            }
            let is_backedge = instr.starts_with("cbnz") && instr.ends_with(self.loop_label);
            let converted = self.convert(instr)?;
            match (in_body, past_backedge) {
                (false, _) => head.push(converted),
                (true, false) => body.push(converted),
                (true, true) => tail.push(converted),
            }
            if is_backedge {
                past_backedge = true;
            }
        }
        if !past_backedge {
            return Err(format!(
                "{}: no back-edge to {} found",
                self.name, self.loop_label
            ));
        }

        let (prologue, preamble) = head.split_at(prologue_len.min(head.len()));

        let mut out = String::new();
        let indent = "        ";
        let mut push_section = |title: &str, lines: &[String], label: Option<&str>| {
            if lines.is_empty() {
                return;
            }
            out.push_str(&format!("{indent}// {title}\n"));
            if let Some(l) = label {
                out.push_str(&format!("    \"{l}\",\n"));
            }
            for l in lines {
                out.push_str(&format!("{indent}\"{l}\",\n"));
            }
            out.push('\n');
        };
        push_section(
            "Prologue: broadcast constants and set the iteration count.",
            prologue,
            None,
        );
        push_section("Preamble — seeds the in-flight iterations.", preamble, None);
        let body_title = match (body_cycles, body_ipc) {
            (Some(cy), Some(ipc)) if cy > 0 => {
                format!("Steady-state body — {cy} cy/iter (IPC {ipc}) on the M4 model.")
            }
            _ => "Steady-state body.".to_string(),
        };
        push_section(&body_title, &body, Some("2:"));
        push_section("Postamble — drains the in-flight iterations.", &tail, None);

        for op in self.operands {
            out.push_str(&format!("{indent}{}\n", op.binding));
        }
        for line in clobber_lines(&[prologue, preamble, &body, &tail])? {
            out.push_str(&format!("{indent}{line}\n"));
        }
        out.push_str(&format!("{indent}options(nostack),\n"));
        Ok(out)
    }

    /// Converts one instruction line: escapes literal braces, substitutes
    /// operand registers, and rewrites the loop back-edge label.
    fn convert(&self, instr: &str) -> Result<String, String> {
        let mut s = instr.replace('{', "{{").replace('}', "}}");
        for op in self.operands {
            s = replace_token(&s, op.reg, op.placeholder);
        }
        s = replace_token(&s, self.loop_label, "2b");
        if s.contains('"') || s.contains('\\') {
            return Err(format!(
                "{}: unexpected quote/backslash in `{instr}`",
                self.name
            ));
        }
        Ok(s)
    }

    /// Regenerates the `asm!` invocation inside `target_fn`. With `write`,
    /// the file is updated; otherwise the region is only compared (for
    /// `check`). Returns whether the on-disk region already matched.
    ///
    /// The region is located syntactically — the unique `fn <target_fn>(`
    /// line, then its `core::arch::asm!(` call, closed by paren matching —
    /// so the production source needs no generation markers.
    fn splice(&self, root: &Path, write: bool) -> Result<bool, String> {
        let target = root.join(self.target_file);
        let text = read(&target)?;
        let lines: Vec<&str> = text.lines().collect();

        let fn_needle = format!("fn {}(", self.target_fn);
        let mut it = lines
            .iter()
            .enumerate()
            .filter(|(_, l)| l.contains(&fn_needle));
        let (fn_idx, _) = it.next().ok_or_else(|| {
            format!(
                "{}: `{fn_needle}` not found in {}",
                self.name, self.target_file
            )
        })?;
        if it.next().is_some() {
            return Err(format!("{}: `{fn_needle}` is not unique", self.name));
        }

        let start = lines[fn_idx..]
            .iter()
            .position(|l| l.trim().starts_with("core::arch::asm!("))
            .map(|offset| fn_idx + offset)
            .ok_or_else(|| format!("{}: no asm! invocation in {}", self.name, self.target_fn))?;
        // Walk to the invocation's closing `);`, counting parens outside
        // string literals (operand bindings like `out("v0")` contribute
        // balanced pairs; template strings contain no parens).
        let mut depth = 0i32;
        let mut end = None;
        'scan: for (i, line) in lines.iter().enumerate().skip(start) {
            let mut in_str = false;
            for c in line.chars() {
                match c {
                    '"' => in_str = !in_str,
                    '(' if !in_str => depth += 1,
                    ')' if !in_str => {
                        depth -= 1;
                        if depth == 0 {
                            end = Some(i);
                            break 'scan;
                        }
                    }
                    _ => {}
                }
            }
        }
        let end = end.ok_or_else(|| format!("{}: unbalanced parens in asm! block", self.name))?;

        let current = lines[start..=end].join("\n");
        let generated = format!("    core::arch::asm!(\n{}    );", self.transcribe(root)?);
        let matches = current == generated;
        if write && !matches {
            let new_text = [&lines[..start], &[generated.as_str()], &lines[end + 1..]]
                .concat()
                .join("\n")
                + "\n";
            fs::write(&target, new_text)
                .map_err(|err| format!("write {}: {err}", target.display()))?;
        }
        Ok(matches)
    }
}

/// The SLOTHY fork checkout: `$SLOTHY_DIR`, defaulting to the nested
/// checkout in the whittle repo (`whittle/slothy`).
fn slothy_dir() -> Result<PathBuf, String> {
    if let Ok(dir) = env::var("SLOTHY_DIR") {
        return Ok(PathBuf::from(dir));
    }
    let home = env::var("HOME").map_err(|_| "neither $SLOTHY_DIR nor $HOME set".to_string())?;
    Ok(PathBuf::from(home).join("src/github.com/selkie-cryptography/whittle/slothy"))
}

fn read(path: &Path) -> Result<String, String> {
    fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))
}

/// The lines strictly between `<fn_label>:` and the closing `ret`.
fn region_lines(text: &str, fn_label: &str) -> Result<Vec<String>, String> {
    let opener = format!("{fn_label}:");
    let mut in_region = false;
    let mut out = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if !in_region {
            in_region = trimmed == opener;
            continue;
        }
        if trimmed == "ret" {
            return Ok(out);
        }
        out.push(line.to_string());
    }
    Err(format!("no `{opener}` .. `ret` region found"))
}

/// Whether a `.s` line is an instruction (not blank, comment, or directive).
fn is_instruction(line: &str) -> bool {
    let t = line.trim();
    !t.is_empty() && !t.starts_with("//") && !t.starts_with('.') && !t.ends_with(':')
}

/// Drops a trailing `// ...` comment from an instruction line.
fn strip_comment(line: &str) -> &str {
    match line.find("//") {
        Some(i) => line[..i].trim_end(),
        None => line.trim_end(),
    }
}

/// Replaces whole-token occurrences of `from` with `to` (a token boundary is
/// any non-alphanumeric, non-underscore character).
fn replace_token(s: &str, from: &str, to: &str) -> String {
    let bytes = s.as_bytes();
    let is_word = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < s.len() {
        let boundary_before = i == 0 || !is_word(bytes[i - 1]);
        let end = i + from.len();
        let boundary_after = end >= s.len() || !is_word(bytes[end]);
        if boundary_before && boundary_after && s[i..].starts_with(from) {
            out.push_str(to);
            i = end;
        } else {
            // Safe to advance one byte: the input is ASCII assembly.
            out.push(bytes[i] as char);
            i += 1;
        }
    }
    out
}

/// Derives the clobber list from every register the converted instructions
/// still mention (operand registers have already been substituted away).
/// Errors if a reserved register (`v8`-`v15`) appears.
fn clobber_lines(sections: &[&[String]]) -> Result<Vec<String>, String> {
    let mut vregs = [false; 32];
    let mut xregs = [false; 32];
    for line in sections.iter().flat_map(|s| s.iter()) {
        for token in line.split(|c: char| !c.is_ascii_alphanumeric()) {
            let Some(first) = token.chars().next() else {
                continue;
            };
            let Ok(n) = token[1..].parse::<usize>() else {
                continue;
            };
            if n >= 32 || token[1..].len() > 2 {
                continue;
            }
            match first {
                'v' | 'q' => vregs[n] = true,
                'x' | 'w' => xregs[n] = true,
                _ => {}
            }
        }
    }
    for (n, used) in vregs.iter().enumerate() {
        if *used && (8..=15).contains(&n) {
            return Err(format!("reserved register v{n} appears in the schedule"));
        }
    }

    let mut lines = Vec::new();
    for (n, used) in xregs.iter().enumerate() {
        if *used {
            lines.push(format!("out(\"x{n}\")  _,"));
        }
    }
    let used: Vec<usize> = (0..32).filter(|&n| vregs[n]).collect();
    for chunk in used.chunks(4) {
        let cells: Vec<String> = chunk
            .iter()
            .map(|n| format!("out(\"v{n}\"){} _,", if *n < 10 { " " } else { "" }))
            .collect();
        lines.push(cells.join(" "));
    }
    Ok(lines)
}

fn usage() -> String {
    let names: Vec<&str> = KERNELS.iter().map(|k| k.name).collect();
    format!(
        "usage: cargo xtask slothy <run|gen> <kernel>... | cargo xtask slothy check\n\
         kernels: {}",
        names.join(", ")
    )
}

fn main() -> ExitCode {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    let args: Vec<String> = env::args().skip(1).collect();
    let (cmd, names) = match args.split_first() {
        Some((sub, rest)) if sub == "slothy" => match rest.split_first() {
            Some((cmd, names)) => (cmd.clone(), names.to_vec()),
            None => {
                eprintln!("{}", usage());
                return ExitCode::FAILURE;
            }
        },
        _ => {
            eprintln!("{}", usage());
            return ExitCode::FAILURE;
        }
    };

    let selected: Vec<&KernelSpec> = if cmd == "check" || names.iter().any(|n| n == "--all") {
        KERNELS.iter().collect()
    } else {
        let mut sel = Vec::new();
        for name in &names {
            match KernelSpec::by_name(name) {
                Some(k) => sel.push(k),
                None => {
                    eprintln!("unknown kernel `{name}`\n{}", usage());
                    return ExitCode::FAILURE;
                }
            }
        }
        sel
    };
    if selected.is_empty() {
        eprintln!("{}", usage());
        return ExitCode::FAILURE;
    }

    let mut failed = false;
    for kernel in selected {
        let result = match cmd.as_str() {
            "run" => kernel
                .run_slothy(&root)
                .and_then(|()| kernel.splice(&root, true))
                .map(|_| format!("{}: scheduled + spliced", kernel.name)),
            "gen" => kernel.splice(&root, true).map(|matched| {
                if matched {
                    format!("{}: up to date", kernel.name)
                } else {
                    format!("{}: spliced", kernel.name)
                }
            }),
            "check" => kernel.splice(&root, false).and_then(|matched| {
                if matched {
                    Ok(format!("{}: up to date", kernel.name))
                } else {
                    Err(
                        "spliced code drifted from archived schedule; run `cargo xtask slothy gen`"
                            .to_string(),
                    )
                }
            }),
            other => {
                eprintln!("unknown command `{other}`\n{}", usage());
                return ExitCode::FAILURE;
            }
        };
        match result {
            Ok(msg) => println!("{msg}"),
            Err(err) => {
                eprintln!("{}: {err}", kernel.name);
                failed = true;
            }
        }
    }
    if failed {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
