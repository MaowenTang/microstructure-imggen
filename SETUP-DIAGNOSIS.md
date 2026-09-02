# SETUP-DIAGNOSIS.md — one-time setup and the round protocol

Architecture (measured constraint, see loop/DECISIONS.md): the Cowork cloud
sandbox cannot reach ACES (port-22 egress blocked), so **Codex runs on the
ACES login node inside this repo**, and Claude stays commander in the cloud.
Files move by two paste commands per round. Bundle members use plain
relative paths (no `./` prefix), so every `tar` command below is exact.
Replace `<REPO_ROOT>` once, everywhere.

## 0. Quick start (download → Codex running T-001)

Run the blocks in order: A on your Mac, B–E on ACES. Block B tells you
whether C applies as-is; blocks that read the tarball must run BEFORE the
`rm` in block D.

```bash
# A — Mac: upload under a fixed remote name (local filename irrelevant)
scp ~/Downloads/winwin_scaffold_v5.tgz aces:winwin_scaffold.tgz
```

```bash
# B — ACES: preflight. Must end with "preflight-done" and print NOTHING else.
cd <REPO_ROOT> \
 && git rev-parse --is-inside-work-tree >/dev/null 2>&1 || echo "NO-GIT: run: git init -b main && tar xzf ~/winwin_scaffold.tgz .gitignore && git add -A && git commit -m init   (then re-run B; the EXISTS: .gitignore it prints is ours — skip that C line)" \
 ;  git rev-parse --is-inside-work-tree >/dev/null 2>&1 && { git rev-parse --verify -q main >/dev/null || echo "NO-MAIN: default branch is $(git rev-parse --abbrev-ref HEAD 2>/dev/null); run: git branch -m main"; } \
 ;  git rev-parse --is-inside-work-tree >/dev/null 2>&1 && { [ "$(git rev-parse --abbrev-ref HEAD 2>/dev/null)" = main ] || echo "NOT-ON-MAIN: run: git checkout main"; } \
 ;  git rev-parse --is-inside-work-tree >/dev/null 2>&1 && { [ -z "$(git status --porcelain 2>/dev/null)" ] || echo "DIRTY: commit first, or: git stash -u"; } \
 ;  for f in $(tar tzf ~/winwin_scaffold.tgz | grep -v "/$"); do [ -e "$f" ] && echo "EXISTS: $f"; done \
 ;  command -v npm >/dev/null || echo "NO-NPM: try: module spider nodejs" \
 ;  echo preflight-done
```

```bash
# C — ACES: collision handling. Run ONLY the lines for EXISTS: entries B printed.
cd <REPO_ROOT>
tar xzf ~/winwin_scaffold.tgz -O .gitignore >> .gitignore                              # EXISTS: .gitignore
printf '\n\n' >> AGENTS.md  && tar xzf ~/winwin_scaffold.tgz -O AGENTS.md  >> AGENTS.md  # EXISTS: AGENTS.md
printf '\n\n' >> CLAUDE.md  && tar xzf ~/winwin_scaffold.tgz -O CLAUDE.md  >> CLAUDE.md  # EXISTS: CLAUDE.md
git mv PROJECT.md PROJECT.repo.md && git commit -m "make room for winwin PROJECT.md"      # EXISTS: PROJECT.md (ours is replaced wholesale on every ACCEPT, so it must own the name)
tar xzf ~/winwin_scaffold.tgz -O README.md > README.winwin.md                           # EXISTS: README.md (optional)
# EXISTS: loop/...  or  SETUP-DIAGNOSIS.md  → an older scaffold is deployed: STOP, use §1b
# EXISTS: checks/.gitkeep, src/.gitkeep, ... → harmless, ignore
```

```bash
# D — ACES: extract (never overwrites; merged files from C are left alone) and commit the scaffold only
cd <REPO_ROOT> && git checkout main && tar xzf ~/winwin_scaffold.tgz --skip-old-files && rm ~/winwin_scaffold.tgz \
 && git add .gitignore AGENTS.md CLAUDE.md PROJECT.md README.md SETUP-DIAGNOSIS.md checks loop results scripts src \
 && git commit -m "winwin scaffold for gap-diagnosis" \
 && grep -c "merge-base" loop/tasks/T-003.md && grep -c "Hard rules" AGENTS.md && ls loop/tasks
```

(Expected tail: two nonzero counts and T-001…T-005 listed. `README.winwin.md`, if created, is intentionally left untracked.)

```bash
# E — ACES: codex once, then dispatch T-001
mkdir -p ~/.npm-global && npm config set prefix ~/.npm-global && npm install -g @openai/codex \
 && echo 'export PATH=$HOME/.npm-global/bin:$PATH' >> ~/.bashrc && source ~/.bashrc && codex login
cd <REPO_ROOT> && codex "Execute task loop/tasks/T-001.md following AGENTS.md. When done, write loop/reports/T-001.md, set the task status to NEEDS_AUDIT, and commit to branch task/T-001."
```

`codex login` is your own auth step; Claude never handles the credential.
If the login node blocks the browser flow, `codex login --api-key <key>` on
ACES works (the key stays on ACES; never paste it into the Claude chat).
Suggested approval mode: on-request until T-001 passes. When Codex
finishes: §3 step 2, attach the bundle in chat, say "审计 T-001".

## 1b. Controlled update over an older scaffold (only before any task is dispatched)

Overwrites commander-owned files only, never executor territory. For every
file you MERGED in block C (AGENTS.md / CLAUDE.md), add `--exclude=<file>`
and re-merge it by hand instead (PROJECT.md was renamed, not merged, so it
is safe to overwrite).

```bash
cd <REPO_ROOT> && tar xzf ~/winwin_scaffold.tgz --exclude=checks --exclude=src --exclude=scripts --exclude=results --exclude=loop/reports --exclude=loop/audits --exclude=README.md --exclude=.gitignore \
 && rm ~/winwin_scaffold.tgz && git add CLAUDE.md PROJECT.md SETUP-DIAGNOSIS.md AGENTS.md loop && git commit -m "winwin scaffold update"
```

(Overwriting `loop/tasks/` resets any `status:` Codex may have set — hence
"before any task is dispatched".)

## 3. The round protocol (fixed order — do not reorder)

Branch ownership: Codex commits only on `task/T-NNN` and leaves the tree
clean. Task-scoped commander files (`loop/audits/T-NNN.md`, the `status:`
line of `loop/tasks/T-NNN.md`) are committed on `task/T-NNN` **before** the
merge so they reach main with it. Project-scoped commander files
(`PROJECT.md`, `loop/DECISIONS.md`, new briefs `loop/tasks/T-MMM.md`) are
committed **only on main, after** the merge — this keeps parallel branches
(T-003 ∥ T-005) from conflicting on PROJECT.md. **Only you advance main.**
SLURM jobs never run code from this shared checkout (AGENTS.md snapshot
rule), so switching branches while a T-004 chain is queued is safe.

| # | Step | Where | Paste |
|---|---|---|---|
| 1 | Dispatch | ACES, repo root | `codex "Execute task loop/tasks/T-NNN.md following AGENTS.md. When done, write loop/reports/T-NNN.md, set the task status to NEEDS_AUDIT, and commit to branch task/T-NNN."` |
| 2 | Pull audit bundle | Mac | `ssh aces 'cd <REPO_ROOT> && git checkout -q task/T-NNN && git log --graph --oneline --decorate --all -40 > loop/GITLOG.txt && git diff main...task/T-NNN > loop/DIFF_T-NNN.patch && tar czf - loop checks scripts src results AGENTS.md CLAUDE.md PROJECT.md' > ~/Downloads/audit_T-NNN.tgz` → attach the .tgz in the Claude chat; say "审计 T-NNN" (AGENTS/CLAUDE ride along so Claude edits them from the real file — including any repo-original section merged in block C — instead of rebuilding them blind; PROJECT.md rides along for reference only — Claude replaces it from its own authoritative state) |
| 3 | Claude rules | cloud | you receive `winwin_update_T-NNN.tgz` (members: `loop/audits/T-NNN.md`, `loop/tasks/T-NNN.md`; on ACCEPT also `PROJECT.md`, `loop/DECISIONS.md`, any new `loop/tasks/T-MMM.md`, and — only when a rule was ratcheted — `AGENTS.md` / `CLAUDE.md`) |
| 4 | Upload update | Mac | `scp ~/Downloads/winwin_update_T-NNN.tgz aces:winwin_update.tgz` |
| 5a | **If REVISE** | ACES, repo root | `git checkout task/T-NNN && tar xzf ~/winwin_update.tgz && git add loop/audits/T-NNN.md loop/tasks/T-NNN.md && git commit -m "audit T-NNN: REVISE" && rm ~/winwin_update.tgz` then rework: `codex "Fix task T-NNN according to loop/audits/T-NNN.md, update the report, set status NEEDS_AUDIT, commit."` → back to step 2 |
| 5b | **If ACCEPT** — task-scoped files onto the task branch, merge, stay on main, project-scoped files onto main | ACES, repo root | `git checkout task/T-NNN && tar xzf ~/winwin_update.tgz loop/audits/T-NNN.md loop/tasks/T-NNN.md && git add loop/audits/T-NNN.md loop/tasks/T-NNN.md && git commit -m "audit T-NNN: ACCEPT" && git checkout main && git merge --no-ff task/T-NNN -m "merge T-NNN (accepted)" && tar xzf ~/winwin_update.tgz && git add PROJECT.md AGENTS.md CLAUDE.md loop && git commit --allow-empty -m "project state after T-NNN" && rm ~/winwin_update.tgz && git rev-parse HEAD` |
| 6 | Relay the hash | chat | paste the `git rev-parse HEAD` output to Claude (it also appears in the next GITLOG.txt); Claude records it in PROJECT.md's verdict column in the next update. The machine gate downstream remains `git merge-base --is-ancestor task/T-NNN main`, not the hash |
| 7 | Dispatch dependents | ACES | only after 5b completed for every prerequisite named in the dependent brief's dispatch gate |

Notes: in 5b the second `tar xzf` re-extracts the two task-scoped files as
identical no-ops and adds the project-scoped ones on main; `--allow-empty`
keeps the chain alive when nothing project-scoped changed. Never append
`git checkout -` to a merge — the following commit must land on main. On
ESCALATE, apply the audit file like 5a but do not dispatch; discuss with
Claude. `loop/GITLOG.txt` and `loop/DIFF_*.patch` are gitignored auditor
scratch. Long training jobs (T-004): Codex sets status `WAITING_HPC` and
exits; when the jobs finish, re-run step 1 with the same command — it
collects results and continues. No terminal needs to stay open.
