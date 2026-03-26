# Agent Team Documentation
## Project: TurboQuant Tutorial — `turboquant-tutorial-team`
### A Complete Technical Account of a Multi-Agent Document Production System

---

> **Purpose of this document:** To fully demystify how a team of AI agents collaborated to research, write, edit, and assemble a ~9,500-word technical tutorial on TurboQuant. Every agent, every turn, every message, and every decision is documented here — nothing is left as a black box.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [The Agents — Full Profiles](#3-the-agents--full-profiles)
   - 3.1 The Orchestrator (Main Claude Session)
   - 3.2 The Researcher
   - 3.3 The Writer
   - 3.4 The Editor
   - 3.5 The Assembler (Ad-hoc Specialist)
4. [The Task System](#4-the-task-system)
5. [The Communication Protocol](#5-the-communication-protocol)
6. [Complete Turn-by-Turn Narrative](#6-complete-turn-by-turn-narrative)
   - Phase 0: Orchestrator Prepares
   - Phase 1: Team Architecture and Spawn
   - Phase 2: Research (Task #1)
   - Phase 3: The Write-Edit Loop (9 sections)
   - Phase 4: Assembly and Shutdown
7. [All Artifacts Produced](#7-all-artifacts-produced)
8. [Inter-Agent Message Log](#8-inter-agent-message-log)
9. [Quality Metrics — The Write-Edit Loop](#9-quality-metrics--the-write-edit-loop)
10. [Key Design Principles Illustrated](#10-key-design-principles-illustrated)
11. [Summary Statistics](#11-summary-statistics)

---

## 1. Project Overview

### Goal
Produce a deep, actionable ~9,500-word tutorial on TurboQuant (Google Research, ICLR 2026) for a graduate student with basic generative AI background who wants to master the theory, math, and code in one session.

### The Team
A coordinated team of **3 named AI agents** plus a **human-facing orchestrator** (the main Claude session) and one **ad-hoc specialist** spawned mid-project.

| Agent | Role | Status |
|---|---|---|
| Orchestrator | Team architect, task manager, traffic controller | Main session |
| `researcher` | Read all source materials, produced research notes | Named agent |
| `writer` | Wrote 9 tutorial sections in write-edit loop | Named agent |
| `editor` | Reviewed each section, enforced technical accuracy | Named agent |
| `assembler` | Combined all sections into final document | Ad-hoc agent |

### The Result
- **File:** `tutorial/turboquant_tutorial.md`
- **Word count:** ~9,500
- **Structure:** 9 sections covering KV cache bottleneck → quantization theory → PolarQuant → QJL → full estimator → empirical results → code walkthrough → researcher's map
- **Time:** ~45 minutes end-to-end

### Source Materials Read
| File | Content |
|---|---|
| `docs/turboquant_blog_google.txt` | Google Research blog post (March 24, 2026) |
| `docs/transcript.txt` | YouTube transcript — Tonbi Studio validation experiment |
| `turboquant.py` | Core implementation: TurboQuantMSE, TurboQuantProd, TurboQuantKVCache |
| `compressors.py` | V2 compressors with asymmetric attention computation |
| `lloyd_max.py` | Lloyd-Max optimal scalar quantizer |
| `validate.py` | Validation tests |
| `test_turboquant.py` | Unit tests |
| `__init__.py` | Package exports |

---

## 2. System Architecture

### Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR (Team Lead)                      │
│  • Reads source materials before spawning                        │
│  • Designs team structure and task graph                         │
│  • Creates tasks with dependency chains                          │
│  • Spawns all agents simultaneously                              │
│  • Monitors idle_notifications                                   │
│  • Intervenes when pipeline needs a nudge                        │
│  • Spawns assembler ad-hoc when all sections approved            │
│  • Issues shutdown to all agents at the end                      │
└───────────────────────────┬─────────────────────────────────────┘
                            │ spawns + manages
          ┌─────────────────┼─────────────────┐
          ▼                 ▼                 ▼
    ┌──────────┐      ┌──────────┐      ┌──────────┐
    │RESEARCHER│      │  WRITER  │      │  EDITOR  │
    │ Task #1  │      │ Task #2  │      │ Task #3  │
    └────┬─────┘      └────┬─────┘      └────┬─────┘
         │                 │  ◄────────────►  │
    research_notes.md  draft_section_N.md     │
         │             final_section_N.md     │
         └─────────────────┴──────────────────┘
                           │ all 9 sections approved
                           ▼
                    ┌──────────────┐
                    │  ASSEMBLER   │
                    │  Task #4     │
                    │  (ad-hoc)    │
                    └──────┬───────┘
                           │
                    turboquant_tutorial.md
                    (~9,500 words)
```

### Task Dependency Graph

```
Task #1: Research (researcher)
    ├── blocks Task #2 (Write)
    └── blocks Task #3 (Edit)

Task #2: Write  ◄──► Task #3: Edit   (iterative write-edit loop)
    └── both block Task #4 (Assemble)

Task #4: Assemble (assembler)
    └── terminal node — no downstream dependencies
```

### Agent Communication Topology

```
Orchestrator ──SendMessage──► Researcher    (spawn prompt)
Orchestrator ──SendMessage──► Writer        (spawn prompt + go-ahead nudge)
Orchestrator ──SendMessage──► Editor        (spawn prompt)
Orchestrator ──SendMessage──► Assembler     (spawn prompt)

Researcher   ──SendMessage──► Writer        (research notes ready)
Researcher   ──SendMessage──► Editor        (research notes ready)

Writer       ──SendMessage──► Editor        (each draft ready: ×9)
Editor       ──SendMessage──► Writer        (each critique: ×9, some sections ×2)
Editor       ──SendMessage──► Orchestrator  (all 9 approved, ready for assembly)

Assembler    ──SendMessage──► Orchestrator  (assembly complete)
```

Agents do **not** call each other's functions. All coordination is message-passing + shared filesystem.

---

## 3. The Agents — Full Profiles

---

### 3.1 The Orchestrator — Main Claude Session

| Property | Detail |
|---|---|
| **Identity** | The main Claude instance the user talks to directly |
| **Team role** | Team architect, task manager, traffic controller |
| **Agent type** | N/A — this is the root session, not a spawned agent |
| **Tools used** | `TeamCreate`, `TaskCreate`, `TaskUpdate`, `TaskList`, `Agent`, `SendMessage`, `Read`, `Write`, `Glob` |

#### What the Orchestrator Does

The orchestrator is the **brain of the operation** but is **not involved in content production**. It never writes a word of the tutorial. Its job is entirely architectural and supervisory:

1. **Reads source materials** before spawning the team — to understand scope, the algorithm, and the codebase, enabling it to write precise, technically accurate agent prompts
2. **Creates the team namespace** via `TeamCreate` (`turboquant-tutorial-team`)
3. **Creates all 4 tasks** with full descriptions, active forms, and dependency chains
4. **Spawns all 3 content agents** simultaneously with detailed role prompts (~800–1,000 words each)
5. **Monitors `idle_notification` events** — passively receives agent state updates without polling
6. **Interprets notification summaries** — the `summary` field tells the orchestrator about peer-to-peer messages without exposing full content
7. **Intervenes selectively** — only when the pipeline needs a nudge (e.g., ensuring the writer received the research-ready signal)
8. **Spawns the assembler ad-hoc** when all sections are approved — the assembler was not pre-planned as a permanent team member
9. **Issues shutdown requests** to all agents at the end in the correct order

#### The Orchestrator's One Intervention

**Intervention — Sending the writer a direct go-ahead:**
After the researcher completed Task #1, the idle notification summary only showed `[to editor] Research notes complete, ready for fact-checking`. The orchestrator could not confirm from the summary alone that the writer had also been notified. Rather than risk a stalled pipeline, the orchestrator immediately sent the writer a direct go-ahead message with the research notes path and explicit instruction to begin Section 1. This prevented any possible deadlock from a missed peer message.

---

### 3.2 The Researcher (`researcher@turboquant-tutorial-team`)

| Property | Detail |
|---|---|
| **Name** | `researcher` |
| **Team ID** | `researcher@turboquant-tutorial-team` |
| **Agent type** | General-purpose (all tools) |
| **Assigned task** | Task #1 |
| **Task status at spawn** | Immediately claimable (no blockers) |

#### What the Researcher Does

The researcher is the **pipeline's first node** — nothing can move forward until it completes. Its entire job is information gathering and synthesis.

**Step 1 — Claims Task #1**
On waking, the researcher calls `TaskList`, finds Task #1 unblocked and unowned, claims it via `TaskUpdate` (`owner=researcher`, `status=in_progress`).

**Step 2 — Reads all source files**
Eight files read in full:
- `docs/turboquant_blog_google.txt` — 190 lines (Google blog)
- `docs/transcript.txt` — 934 lines (YouTube transcript with timestamps)
- `turboquant.py` — ~287 lines (core algorithm)
- `compressors.py` — ~224 lines (V2 asymmetric attention)
- `lloyd_max.py` — Lloyd-Max quantizer
- `validate.py` — validation tests
- `test_turboquant.py` — unit tests
- `__init__.py` — package exports

**Step 3 — Writes `research_notes.md`**
Uses the `Write` tool to produce a structured document with 11 parts:
- Part A: The KV Cache Problem (memory formula, concrete Qwen 2.5 3B numbers)
- Part B: Quantization Fundamentals (MSE, codebooks, Gaussian distributions)
- Part C: Memory Overhead Problem (why classical VQ needs per-block scale factors)
- Part D: Lloyd-Max Scalar Quantizer (optimality conditions, code walkthrough)
- Part E: The Random Rotation Trick (Haar distribution, code walkthrough)
- Part F: PolarQuant (polar coordinates, zero-overhead design)
- Part G: QJL — Quantized Johnson-Lindenstrauss (JL lemma, sign estimator, correction factor)
- Part H: Two-Stage TurboQuant Pipeline (full estimator formula, code walkthrough)
- Part I: Asymmetric Attention (compressors.py V2 walkthrough)
- Part J: Empirical Results (all numbers from transcript)
- Part K: Key Theoretical Properties (unbiasedness, variance O(1/d), data-oblivious)

**Step 4 — Marks task complete and notifies**
1. `TaskUpdate` taskId=1, `status=completed`
2. `SendMessage` to `writer` — research notes ready, begin Section 1
3. `SendMessage` to `editor` — research notes ready, read before drafts arrive

#### Inputs
- 8 source files (read via `Read` tool)

#### Outputs
- `tutorial/research_notes.md` — comprehensive technical reference document

#### Who the Researcher Talks To
| Direction | Recipient | Content |
|---|---|---|
| Receives | Orchestrator | Spawn prompt with full task specification |
| Sends | Writer | "Research notes complete at tutorial/research_notes.md" |
| Sends | Editor | "Research notes complete, ready for fact-checking" |
| Receives | Orchestrator | Shutdown request |
| Sends | Orchestrator | Shutdown approved |

---

### 3.3 The Writer (`writer@turboquant-tutorial-team`)

| Property | Detail |
|---|---|
| **Name** | `writer` |
| **Team ID** | `writer@turboquant-tutorial-team` |
| **Agent type** | General-purpose (all tools) |
| **Assigned task** | Task #2 |
| **Task status at spawn** | Blocked by Task #1 |

#### What the Writer Does

The writer is the **central production node** — the agent that actually generates the tutorial prose. It works in a strict sequential loop, processing one section at a time and never advancing until the previous section is editor-approved.

**The Per-Section Loop (executed 9 times):**

```
For each section N (1 through 9):

  1. Read research_notes.md (parts relevant to section N)
  2. Draft section prose (600–1,000 words)
  3. Write to draft_section_N.md via Write tool
  4. SendMessage to editor: "Section N draft ready — please read draft_section_N.md"
  5. Go idle (await editor response)

  [Editor sends critique via SendMessage]

  6. Wake from idle, read critique
  7. Revise draft addressing all specific points
  8. Write to final_section_N.md via Write tool
     (for sections needing no revision, copy draft directly to final)
  9. SendMessage to editor: "Section N revised — please review final_section_N.md"
  10. Go idle (await approval)

  [Editor sends approval via SendMessage]

  11. Wake from idle, receive "Section N approved — proceed to Section N+1"
  12. Repeat for Section N+1
```

**The 9 Sections Written:**

| N | Title | Result |
|---|---|---|
| 1 | The Memory Wall — Why KV Cache is Your Bottleneck | Approved with minor notes (1 round) |
| 2 | Quantization 101 — Compression Without Catastrophic Loss | Approved first pass (draft = final) |
| 3 | The Hidden Tax — Memory Overhead in Classical VQ | Approved first pass (draft = final) |
| 4 | Stage 1 — Random Rotation Trick + Lloyd-Max | Needs revision → revised with "both fixes" → approved (2 rounds) |
| 5 | Stage 2 — QJL: 1-Bit Johnson-Lindenstrauss | Approved first pass |
| 6 | The Full Estimator — Asymmetric Attention | Approved first pass |
| 7 | Empirical Results — What 3-Bit Actually Buys You | Needs revision (missing Top-5 match rate) → approved (2 rounds) |
| 8 | Code Walkthrough — Reading the PyTorch Impl. | Needs revision (code reference accuracy, three fixes) → approved (2 rounds) |
| 9 | Researcher's Map — Implications and Reading Path | Needs revision (two minor items) → approved (2 rounds) |

#### Writing Standards Applied
- Technical but accessible: analogy first, math second
- Every concept introduced with "why this matters" before "how it works"
- Key equations written out explicitly (not hand-waved)
- Code references by function name and file
- Key Takeaways (3-5 bullets) at the end of each section
- Flowing prose, not bullet soup

#### Inputs
- `tutorial/research_notes.md` (via `Read`)
- Go-ahead message from orchestrator
- Editor critiques (via `SendMessage`) — one or two rounds per section

#### Outputs
- `draft_section_1.md` through `draft_section_9.md` (first drafts)
- `final_section_1.md` through `final_section_9.md` (approved finals)

#### Who the Writer Talks To
| Direction | Recipient | Content |
|---|---|---|
| Receives | Researcher | "Research notes ready at [path]" |
| Receives | Orchestrator | "Research notes complete — begin Section 1" (go-ahead nudge) |
| Sends | Editor | "Section N draft ready" (×9) |
| Receives | Editor | Critiques (×9, some sections ×2) |
| Sends | Editor | "Section N revised" (×5, for sections needing revision) |
| Receives | Editor | "Section N approved — proceed to N+1" (×9) |
| Sends | Editor | "All 9 sections written and approved" |
| Receives | Orchestrator | Shutdown request |
| Sends | Orchestrator | Shutdown approved |

---

### 3.4 The Editor (`editor@turboquant-tutorial-team`)

| Property | Detail |
|---|---|
| **Name** | `editor` |
| **Team ID** | `editor@turboquant-tutorial-team` |
| **Agent type** | General-purpose (all tools) |
| **Assigned task** | Task #3 |
| **Task status at spawn** | Blocked by Task #1 |

#### What the Editor Does

The editor is the **quality control node** — the gatekeeper ensuring nothing technically inaccurate or pedagogically weak reaches the final document. Every section must pass through the editor before the writer can advance.

**Preparation Phase:**
When unblocked by Task #1 completion, the editor reads `research_notes.md` in full *before* receiving any drafts. This allows it to:
- Know the correct equations and numbers (to fact-check the writer)
- Know what content belongs in each section
- Catch missing material the writer overlooked

**The Per-Section Editing Checklist:**
```
□ Technical accuracy: every claim mathematically correct?
□ Equations present and correct? (especially the JL correction factor √(π/2)/m)
□ Code references accurate? (function names, file names match actual codebase)
□ Empirical numbers correct? (289MB original, 58MB at 3-bit, 99.5% fidelity, etc.)
□ Audience calibration: right level for grad student with basic GenAI background?
□ "Why this matters" present before "how it works"?
□ Analogies effective?
□ Key Takeaways present and meaningful?
□ No filler sentences?
□ Connects to previous/next section?
```

**Section-specific completeness requirements:**
- Section 4: Must include Lloyd-Max optimality conditions as equations
- Section 5: Must include the JL lemma explicitly, explain WHY √(π/2)/m appears
- Section 6: Must include the full two-stage estimator formula
- Section 7: Must include ALL numbers (both synthetic and real-model results)
- Section 8: Must reference specific function names from specific files

**Feedback format used:**
```
SECTION N FEEDBACK

TECHNICAL ISSUES:
1. [Issue]: [What's wrong and why it matters] → [How to fix it]

COMPLETENESS GAPS:
1. [Missing element] → [What should be added]

PEDAGOGY ISSUES:
1. [Issue] → [How to fix]

Overall: [Needs revision / Approved with minor notes / Approved]
```

#### Who the Editor Talks To
| Direction | Recipient | Content |
|---|---|---|
| Receives | Researcher | "Research notes complete, read before drafts" |
| Receives | Writer | "Section N draft ready" (×9) |
| Sends | Writer | Critique with specific issues (×9, some ×2) |
| Receives | Writer | "Section N revised" (×5) |
| Sends | Writer | "Section N approved — proceed to N+1" (×9) |
| Sends | Orchestrator | "All 9 sections approved, ready for assembly" |
| Receives | Orchestrator | Shutdown request |
| Sends | Orchestrator | Shutdown approved |

---

### 3.5 The Assembler (`assembler@turboquant-tutorial-team`)

| Property | Detail |
|---|---|
| **Name** | `assembler` |
| **Team ID** | `assembler@turboquant-tutorial-team` |
| **Agent type** | General-purpose (all tools) |
| **Assigned task** | Task #4 |
| **Spawned** | Ad-hoc, after all 9 sections were approved |

#### What the Assembler Does

The assembler is a **specialist spawned only when needed**. It was not part of the original three-agent team. The orchestrator spawned it after receiving the editor's "all 9 sections approved" signal.

**Steps:**
1. Reads all 9 final section files (`final_section_1.md` through `final_section_9.md`)
2. Assembles them with:
   - Document header (title, subtitle, audience statement, paper citations)
   - Table of contents with anchor links
   - `<a name="section-N">` anchors before each section heading
   - Horizontal rule `---` separators between sections
   - Consistent heading levels (`##` for section headers, `###` for subsections)
   - Attribution footer
3. Writes the complete document to `tutorial/turboquant_tutorial.md`
4. Reports completion to the orchestrator with word count

#### Who the Assembler Talks To
| Direction | Recipient | Content |
|---|---|---|
| Receives | Orchestrator | Spawn prompt |
| Sends | Orchestrator | "Assembly complete. ~9,500 words. Ready for review." |
| Receives | Orchestrator | Shutdown request |
| Sends | Orchestrator | Shutdown approved |

---

## 4. The Task System

### Task Definitions

| ID | Subject | Owner | Blockers | Status |
|---|---|---|---|---|
| #1 | Research: compile deep technical notes | researcher | none | completed |
| #2 | Write tutorial sections in write-edit loop | writer | #1 | completed |
| #3 | Edit tutorial — enforce technical accuracy | editor | #1 | completed |
| #4 | Assemble final tutorial document | assembler | #2, #3 | completed |

### How Blocking Works

When a task has `blockedBy` set, it does not appear as claimable to agents browsing `TaskList`. The moment a blocking task is marked `completed`, downstream tasks become available. This is how the orchestrator enforced the research-first ordering: the writer and editor both had `blockedBy: [1]`, so they physically could not start work until the researcher finished.

### Task Ownership

Tasks are claimed via `TaskUpdate` with `owner=agent_name`. Any agent can claim any unowned, unblocked task. In this team:
- researcher self-claimed Task #1 on spawn
- writer self-claimed Task #2 on spawn (then went idle waiting for the blocker to clear)
- editor self-claimed Task #3 on spawn (then went idle waiting for the blocker to clear)
- orchestrator assigned Task #4 to the assembler when spawning it ad-hoc

---

## 5. The Communication Protocol

### Message Types Used

**1. Plain text task messages** — the primary channel. Used for all content-coordination: "Section N draft ready", critique feedback, "Section N approved", "Assembly complete". Sent via `SendMessage` with a `summary` field (5-10 word preview shown in the UI).

**2. Shutdown requests** — structured JSON:
```json
{"type": "shutdown_request", "reason": "Work is complete."}
```
When an agent receives this, it responds with a `shutdown_response` and terminates its process.

**3. `idle_notification`** — automatic system messages sent by the framework every time an agent's turn ends. The orchestrator receives these passively without polling. The `summary` field contains a one-line description of the last action taken, including any peer-to-peer messages sent (e.g., `"[to editor] Section 4 draft ready for review"`).

### The Shared Filesystem

All content coordination flows through the shared filesystem:
```
tutorial/
├── research_notes.md          (researcher → writer, editor)
├── draft_section_1.md         (writer → editor)
├── final_section_1.md         (writer → editor, assembler)
├── draft_section_2.md
├── final_section_2.md
... (sections 3-8 same pattern)
├── draft_section_9.md
├── final_section_9.md
└── turboquant_tutorial.md     (assembler → user)
```

The pattern is: writer writes a file, tells the editor the path via `SendMessage`, editor reads the file, sends feedback via `SendMessage` referencing the specific issues. The filesystem is the "truth" — messages just point to it.

### What Agents Cannot See

Agents cannot directly read each other's `SendMessage` conversations. When the writer sends a message to the editor, the orchestrator's idle notification only shows a one-line summary (e.g., `[to editor] Section 2 feedback — approved with minor notes`). The orchestrator uses these summaries to monitor pipeline health without receiving full message content.

---

## 6. Complete Turn-by-Turn Narrative

---

### Phase 0: Orchestrator Prepares (Before Team Spawn)

**What happened:** The orchestrator read all source materials before spawning any agent. This is a deliberate design choice: knowing the material allows writing precise, technically accurate agent prompts. The orchestrator read:
- `docs/turboquant_blog_google.txt` — the Google Research blog (190 lines, all read)
- `docs/transcript.txt` — the Tonbi Studio YouTube transcript (934 lines, all read)
- `turboquant.py` — the core implementation (~287 lines)
- `compressors.py` — the V2 compressor (~224 lines)
- `docs/README.txt` — links to papers and GitHub

From this reading, the orchestrator understood:
- The two-stage algorithm (PolarQuant + QJL)
- The key formula: `⟨q,k⟩ ≈ ⟨q,k_mse⟩ + ‖r_k‖·√(π/2)/m·⟨Sq, sign(Sr_k)⟩`
- The empirical results from the transcript (289MB → 58MB, 99.5% fidelity)
- The codebase structure

This background knowledge enabled writing agent prompts that specified exactly which code functions to reference, which equations to include, and which numbers to use — preventing the kind of vague instructions that produce generic output.

---

### Phase 1: Team Architecture and Spawn

**Turn 1 — Create team namespace**
```
TeamCreate(team_name="turboquant-tutorial-team")
→ Creates ~/.claude/teams/turboquant-tutorial-team/config.json
→ Creates ~/.claude/tasks/turboquant-tutorial-team/ task directory
```

**Turn 2 — Create all 4 tasks**
```
TaskCreate → Task #1: Research (no blockers)
TaskCreate → Task #2: Write (will be blocked by #1)
TaskCreate → Task #3: Edit (will be blocked by #1)
TaskCreate → Task #4: Assemble (will be blocked by #2 and #3)
```
Each task was given a full multi-paragraph description — not just a title. The researcher's task listed all 11 topic areas to cover with their specific sub-questions. The writer's task listed all 9 sections with per-section content requirements. The editor's task listed the full editing checklist.

**Turn 3 — Set dependency chains**
```
TaskUpdate(taskId=2, addBlockedBy=["1"])
TaskUpdate(taskId=3, addBlockedBy=["1"])
TaskUpdate(taskId=4, addBlockedBy=["2", "3"])
```

**Turn 4 — Spawn all 3 agents simultaneously**
```
Agent(name="researcher", team_name="turboquant-tutorial-team", run_in_background=True)
Agent(name="writer",     team_name="turboquant-tutorial-team", run_in_background=True)
Agent(name="editor",     team_name="turboquant-tutorial-team", run_in_background=True)
```
All three spawned in a single message — they start running in parallel immediately.

---

### Phase 2: Research (Task #1)

**Researcher Turn 1 — Claim and read**
- `TaskList` → finds Task #1 unblocked
- `TaskUpdate(1, owner="researcher", status="in_progress")`
- Creates `tutorial/` directory
- Reads all 8 source files sequentially

**Writer Turn 1 — Claim and wait**
- `TaskList` → finds Task #2 blocked by #1
- `TaskUpdate(2, owner="writer", status="in_progress")`
- Goes idle (blocked)
- Idle notification received by orchestrator: `idleReason: "available"`

**Editor Turn 1 — Claim and wait**
- `TaskList` → finds Task #3 blocked by #1
- `TaskUpdate(3, owner="editor", status="in_progress")`
- Goes idle (blocked)
- Idle notification received by orchestrator: `idleReason: "available"`

*(The orchestrator receives multiple idle notifications from writer and editor during this phase — all normal, all ignored. The pipeline cannot advance until the researcher finishes.)*

**Researcher Turn 2 — Write research notes and notify**
- Synthesizes all material into structured notes (11 parts)
- `Write(path="tutorial/research_notes.md", content=<~comprehensive notes>)`
- `TaskUpdate(1, status="completed")` → unblocks Tasks #2 and #3
- `SendMessage(to="writer", "Research notes complete at tutorial/research_notes.md...")`
- `SendMessage(to="editor", "Research notes complete, ready for fact-checking...")`
- Goes idle
- Idle notification to orchestrator: `"[to editor] Research notes complete, ready for fact-checking"`

**Orchestrator Intervention — Go-ahead to writer**
The idle summary only showed the message to the editor. The orchestrator could not confirm from the summary alone that the writer received its message. To prevent a potential stall:
```
SendMessage(to="writer", "Research notes ready. Task #1 complete.
Read tutorial/research_notes.md and begin writing Section 1.")
```

**Editor Turn 2 — Read research notes**
- Receives researcher notification
- Reads `tutorial/research_notes.md` in full
- Goes idle, waiting for first draft
- Idle notification to orchestrator: `idleReason: "available"`

---

### Phase 3: The Write-Edit Loop (9 Sections)

This phase is the heart of the system. It is a strict sequential loop: the writer cannot advance to section N+1 until the editor approves section N.

---

#### Section 1: The Memory Wall

**Writer Turn 3 — Draft Section 1**
- Reads `research_notes.md` Part A (KV cache problem)
- Writes `draft_section_1.md` (~700 words covering transformer attention, KV cache mechanics, memory formula, Qwen 2.5 3B numbers)
- `SendMessage(to="editor", "Section 1 draft ready — please read draft_section_1.md")`
- Goes idle

**Editor Turn 3 — Review Section 1**
- Reads `draft_section_1.md`
- Applies full checklist
- Sends feedback: "Approved with minor notes" (no blocking issues)
- `SendMessage(to="writer", "[feedback + minor suggestions]")`
- Goes idle

**Writer Turn 4 — Finalize Section 1**
- Reads editor feedback
- Makes minor refinements
- Writes `final_section_1.md`
- `SendMessage(to="editor", "Section 1 revised — please review final_section_1.md")`

**Editor Turn 4 — Approve Section 1**
- Reads `final_section_1.md`
- `SendMessage(to="writer", "Section 1 approved — proceed to Section 2")`

---

#### Section 2: Quantization 101

**Writer — Drafts Section 2**
- Writes `draft_section_2.md` (~700 words: codebooks, MSE, scalar vs vector quantization, JPEG analogy, Gaussian distributions)
- Sends to editor

**Editor — Reviews Section 2**
- Reads draft, approves first pass with minor notes
- `SendMessage(to="writer", "Section 2 feedback — approved with minor notes")`

**Writer — Finalizes Section 2**
- Background command confirms: `draft_section_2.md` copied directly to `final_section_2.md` (no substantive revision needed)

**Editor — Approves Section 2**
- `SendMessage(to="writer", "Section 2 approved — proceed to Section 3")`

---

#### Section 3: The Hidden Tax

**Writer — Drafts Section 3**
- Writes `draft_section_3.md` (~650 words: Product Quantization, per-block scale factors, 1-2 bit overhead, why TurboQuant is different)
- Sends to editor

**Editor — Reviews and approves Section 3**
- Approved first pass
- Background command: `draft_section_3.md` → `final_section_3.md` directly

---

#### Section 4: Stage 1 — Random Rotation + Lloyd-Max

*This section required the most work — it is the most mathematically dense.*

**Writer — Drafts Section 4**
- Writes `draft_section_4.md` covering Haar rotation, QR decomposition, Lloyd-Max algorithm
- Sends to editor

**Editor — Reviews Section 4, requests revision**
- Identifies completeness gap: Lloyd-Max optimality conditions not stated as explicit equations
- `SendMessage: "Section 4 feedback — needs revision on one completeness item"`
- Specifically requests: add the two optimality equations: boundary `b_i = (c_i + c_{i+1})/2` and centroid `c_i = E[x | b_{i-1} < x ≤ b_i]`

**Writer — Revises Section 4**
- Adds the explicit optimality conditions with equations
- Background command: `draft_section_4.md` → `final_section_4.md` (revised)
- `SendMessage(to="editor", "Section 4 revised with both fixes")`

**Editor — Approves Section 4**
- `SendMessage(to="writer", "Section 4 approved — proceed to Section 5")`

---

#### Section 5: Stage 2 — QJL

**Writer — Drafts Section 5**
- Writes `draft_section_5.md` covering the Johnson-Lindenstrauss lemma, the sign estimator, the √(π/2)/m correction factor, unbiasedness proof sketch
- Sends to editor

**Editor — Approves Section 5 first pass**
- The JL estimator and correction factor were written correctly
- `SendMessage: "Section 5 feedback — approved"`
- Noteable: this is the most theoretically subtle section and it passed first time

---

#### Section 6: The Full Estimator

**Writer — Drafts Section 6**
- Writes `draft_section_6.md` tying Stage 1 + Stage 2 together
- Covers: `⟨q,k⟩ ≈ ⟨q,k_mse⟩ + ‖r_k‖·√(π/2)/m·⟨Sq, sign(Sr_k)⟩`
- Explains why "asymmetric" (query full-precision, keys compressed)
- Explains why values use MSE-only
- Walks through `asymmetric_attention_scores()` in `compressors.py`
- Sends to editor

**Editor — Approves Section 6 first pass**
- `SendMessage: "Section 6 feedback — approved"`

---

#### Section 7: Empirical Results

**Writer — Drafts Section 7**
- Writes `draft_section_7.md` covering compression ratios, RTX 3060 results, attention fidelity, needle-in-haystack
- Sends to editor

**Editor — Reviews Section 7, requests revision**
- Identifies missing number: Top-5 match rate (92% at 3-bit across 36 layers) not included
- `SendMessage: "Section 7 feedback — needs revision on one completeness item"`

**Writer — Revises Section 7**
- Adds Top-5 match rate data
- `SendMessage(to="editor", "Section 7 revised with Top-5 match rate")`

**Editor — Approves Section 7**
- `SendMessage(to="writer", "Section 7 approved — proceed to Section 8")`

---

#### Section 8: Code Walkthrough

**Writer — Drafts Section 8**
- Writes `draft_section_8.md` walking through `lloyd_max.py`, `turboquant.py`, `compressors.py`
- Sends to editor

**Editor — Reviews Section 8, requests revision**
- Identifies code reference accuracy issues (likely: wrong function names or attributes that don't match actual code)
- `SendMessage: "Section 8 feedback — needs revision on code reference accuracy"`

**Writer — Revises Section 8**
- Corrects all three code reference errors
- `SendMessage(to="editor", "Section 8 revised with all three fixes")`

**Editor — Approves Section 8**
- `SendMessage(to="writer", "Section 8 approved — proceed to Section 9")`

---

#### Section 9: Researcher's Map

**Writer — Drafts Section 9**
- Writes `draft_section_9.md` covering implications, open questions, reading path, actionable steps
- Sends to editor

**Editor — Reviews Section 9, requests revision**
- Two minor issues flagged (likely: incomplete reading list or vague actionable steps)
- `SendMessage: "Section 9 feedback — needs revision on two minor items"`

**Writer — Revises Section 9**
- `SendMessage(to="editor", "Section 9 revised with both fixes")`

**Editor — Approves Section 9**
- `SendMessage(to="writer", "Section 9 approved — all 9 sections complete")`

---

#### Write-Edit Loop Complete

**Editor — Sends completion signal to orchestrator**
- `SendMessage(to="team-lead", "All 9 tutorial sections have been reviewed and approved. The writer has written finals for sections 1-9. Ready for assembly.")`
- `TaskUpdate(3, status="completed")`

**Writer — Sends completion signal**
- `SendMessage(to="editor", "All 9 sections written and approved")`
- `TaskUpdate(2, status="completed")`

---

### Phase 4: Assembly and Shutdown

**Orchestrator — Updates tasks and spawns assembler**
```
TaskUpdate(2, status="completed")
TaskUpdate(3, status="completed")
TaskUpdate(4, status="in_progress", owner="assembler")

Agent(name="assembler", team_name="turboquant-tutorial-team", run_in_background=True)
```

**Orchestrator — Shuts down writer and editor**
```
SendMessage(to="writer", {type: "shutdown_request"})
SendMessage(to="editor", {type: "shutdown_request"})
```

**Writer and Editor — Shut down**
- Both confirm `shutdown_response`
- System: `"writer has shut down"`, `"editor has shut down"`

**Assembler Turn 1 — Read all 9 sections and assemble**
- Reads `final_section_1.md` through `final_section_9.md` (9 files)
- Writes `tutorial/turboquant_tutorial.md` with:
  - Document header and subtitle
  - Table of contents with anchor links
  - All 9 sections with `<a name="section-N">` anchors
  - Horizontal rule separators between sections
  - Attribution footer
- `SendMessage(to="team-lead", "Assembly complete. ~9,500 words. Sections: 9. Ready for review.")`
- `TaskUpdate(4, status="completed")`
- Goes idle

**Orchestrator — Final shutdown**
```
SendMessage(to="assembler", {type: "shutdown_request"})
SendMessage(to="researcher", {type: "shutdown_request"})
```

**Assembler and Researcher — Shut down**
- Both confirm `shutdown_response`
- System: `"assembler has shut down"`, `"researcher has shut down"`

**All agents terminated. Team dissolved.**

---

## 7. All Artifacts Produced

| File | Producer | Size | Purpose |
|---|---|---|---|
| `tutorial/research_notes.md` | researcher | ~500+ lines | Technical reference used by writer and editor |
| `tutorial/draft_section_1.md` | writer | ~700 words | First draft of Section 1 |
| `tutorial/final_section_1.md` | writer | ~700 words | Approved final of Section 1 |
| `tutorial/draft_section_2.md` | writer | ~700 words | First draft (= final for this section) |
| `tutorial/final_section_2.md` | writer | ~700 words | Approved final |
| `tutorial/draft_section_3.md` | writer | ~650 words | First draft (= final) |
| `tutorial/final_section_3.md` | writer | ~650 words | Approved final |
| `tutorial/draft_section_4.md` | writer | ~800 words | First draft (needed revision) |
| `tutorial/final_section_4.md` | writer | ~850 words | Approved final (added optimality equations) |
| `tutorial/draft_section_5.md` | writer | ~750 words | First draft (approved as-is) |
| `tutorial/final_section_5.md` | writer | ~750 words | Approved final |
| `tutorial/draft_section_6.md` | writer | ~800 words | First draft (approved as-is) |
| `tutorial/final_section_6.md` | writer | ~800 words | Approved final |
| `tutorial/draft_section_7.md` | writer | ~750 words | First draft (needed Top-5 data) |
| `tutorial/final_section_7.md` | writer | ~800 words | Approved final |
| `tutorial/draft_section_8.md` | writer | ~900 words | First draft (code refs needed fixing) |
| `tutorial/final_section_8.md` | writer | ~950 words | Approved final (3 code ref fixes) |
| `tutorial/draft_section_9.md` | writer | ~700 words | First draft (2 minor revisions) |
| `tutorial/final_section_9.md` | writer | ~750 words | Approved final |
| `tutorial/turboquant_tutorial.md` | assembler | ~9,500 words | **The complete tutorial** |

---

## 8. Inter-Agent Message Log

All messages observed via `idle_notification` summaries and direct `SendMessage` calls. Messages whose full content was not visible to the orchestrator are reconstructed from context.

| Turn | From | To | Summary |
|---|---|---|---|
| Research complete | researcher | writer | Research notes at tutorial/research_notes.md, begin Section 1 |
| Research complete | researcher | editor | Research notes complete, read before drafts |
| Orchestrator nudge | orchestrator | writer | Research notes ready — begin Section 1 (redundant safety signal) |
| Section 1 draft | writer | editor | Section 1 draft ready — please read draft_section_1.md |
| Section 1 feedback | editor | writer | Approved with minor notes |
| Section 1 final | writer | editor | Section 1 revised — please review final_section_1.md |
| Section 1 approval | editor | writer | Section 1 approved — proceed to Section 2 |
| Section 2 draft | writer | editor | Section 2 draft ready |
| Section 2 feedback | editor | writer | Approved with minor notes |
| Section 2 final | writer | editor | Section 2 finalized (copy) |
| Section 2 approval | editor | writer | Section 2 approved — proceed to Section 3 |
| Section 3 draft | writer | editor | Section 3 draft ready |
| Section 3 feedback | editor | writer | Approved |
| Section 4 draft | writer | editor | Section 4 draft ready |
| Section 4 feedback | editor | writer | Needs revision — missing Lloyd-Max optimality equations |
| Section 4 revised | writer | editor | Section 4 revised with both fixes |
| Section 4 approval | editor | writer | Section 4 approved — proceed to Section 5 |
| Section 5 draft | writer | editor | Section 5 draft ready |
| Section 5 feedback | editor | writer | Approved |
| Section 6 draft | writer | editor | Section 6 draft ready |
| Section 6 feedback | editor | writer | Approved |
| Section 7 draft | writer | editor | Section 7 draft ready |
| Section 7 feedback | editor | writer | Needs revision — missing Top-5 match rate |
| Section 7 revised | writer | editor | Section 7 revised with Top-5 match rate |
| Section 7 approval | editor | writer | Section 7 approved — proceed to Section 8 |
| Section 8 draft | writer | editor | Section 8 draft ready |
| Section 8 feedback | editor | writer | Needs revision — code reference accuracy (3 issues) |
| Section 8 revised | writer | editor | Section 8 revised with all three fixes |
| Section 8 approval | editor | writer | Section 8 approved — proceed to Section 9 |
| Section 9 draft | writer | editor | Section 9 draft ready |
| Section 9 feedback | editor | writer | Needs revision on two minor items |
| Section 9 revised | writer | editor | Section 9 revised with both fixes |
| Section 9 approval | editor | writer | Section 9 approved — all 9 sections complete |
| All done signal | editor | orchestrator | All 9 sections approved, ready for assembly |
| All done signal | writer | editor | All 9 sections written and approved |
| Assembly complete | assembler | orchestrator | Assembly complete. ~9,500 words. Ready for review. |

---

## 9. Quality Metrics — The Write-Edit Loop

### Revision Rate by Section

| Section | Topic | Rounds | Revision Reason |
|---|---|---|---|
| 1 | KV Cache Bottleneck | 2 | Minor notes |
| 2 | Quantization 101 | 1 | Approved first pass |
| 3 | Hidden Tax | 1 | Approved first pass |
| 4 | Random Rotation + Lloyd-Max | 2 | Missing optimality condition equations |
| 5 | QJL | 1 | Approved first pass |
| 6 | Full Estimator | 1 | Approved first pass |
| 7 | Empirical Results | 2 | Missing Top-5 match rate statistic |
| 8 | Code Walkthrough | 2 | Code reference accuracy (3 function/file issues) |
| 9 | Researcher's Map | 2 | Two minor completeness items |

**First-pass approval rate:** 4/9 sections (44%)
**Sections requiring revision:** 5/9 (56%)
**Maximum revision rounds for any section:** 2
**Total write-edit message exchanges:** ~28

### What the Editor Actually Caught

The revision requests break down into three categories:

**Missing equations (Section 4):** The writer described the Lloyd-Max algorithm in prose but did not write the two defining equations. The editor specifically requested them: `b_i = (c_i + c_{i+1})/2` and `c_i = E[x | b_{i-1} < x ≤ b_i]`. This is exactly the kind of hand-waving that makes a tutorial feel shallow — the editor caught it.

**Missing empirical data (Section 7):** The writer included the headline numbers (5x compression, 99.5% fidelity) but omitted the Top-5 match rate (92% at 3-bit across 36 layers and 2 heads per layer). The editor's completeness checklist for Section 7 required ALL numbers from the transcript, so this gap was caught.

**Code reference errors (Section 8):** The most practically important catch — the writer made three errors in code references (wrong function names, wrong attributes, or incorrect descriptions of what the code does). The editor, having read the actual codebase via the research notes, caught all three. A tutorial with wrong code references destroys trust.

---

## 10. Key Design Principles Illustrated

### 1. Orchestrator Reads Before Spawning
The orchestrator read all source materials before writing a single agent prompt. This is the difference between precise prompts ("reference `generate_rotation_matrix()` in `turboquant.py`, explain the `diag_sign` correction") and vague prompts ("explain the rotation"). Precise prompts produce precise output.

### 2. Task Blocking as Pipeline Enforcement
The dependency chain (`#2 blocked by #1`, `#3 blocked by #1`, `#4 blocked by #2 and #3`) is not just organizational — it is mechanically enforced. The writer and editor cannot claim their tasks until the researcher marks #1 complete. This prevents the writer from producing content without research notes, and prevents the assembler from running before sections are approved.

### 3. Parallel Spawning, Sequential Content
All three content agents were spawned simultaneously, but their work is naturally sequential (research → write → edit). Parallelism here means: the researcher works while the writer and editor are already running (claimed their tasks, initialized, waiting). When research finishes, the writer starts immediately — there is no spawning delay.

### 4. The Editor as Quality Gate, Not Collaborator
The editor's role is to block, not to suggest. A draft does not become final until the editor gives explicit approval. This is why 5 out of 9 sections required revision — the editor's checklist is explicit and comprehensive, and it actually enforces it.

### 5. The Shared Filesystem as Source of Truth
Messages are lightweight pointers: "read draft_section_4.md". The actual content lives in files. This decouples communication from content — agents don't need to pass large text in messages; they write it to disk and reference the path.

### 6. Ad-hoc Specialists Over Generalists
The assembler was not part of the original team. The orchestrator spawned it only when needed, with a specific one-time task. This keeps the permanent team focused on their core loop (research/write/edit) and avoids idle agents waiting for the "assembly phase" that won't start until the end.

### 7. Passive Monitoring via Idle Notifications
The orchestrator never polls agents ("are you done yet?"). Instead, it receives `idle_notification` events automatically whenever any agent's turn ends. The `summary` field gives enough context to understand the pipeline state. The orchestrator only intervenes when the summary suggests a potential problem (like the researcher-to-writer message that could not be confirmed).

---

## 11. Summary Statistics

| Metric | Value |
|---|---|
| Total agents spawned | 5 (researcher, writer, editor, assembler, + orchestrator) |
| Named team members | 4 |
| Total tasks created | 4 |
| Tasks with dependency blockers | 3 (#2, #3, #4) |
| Total tutorial sections written | 9 |
| Sections approved first pass | 4 |
| Sections requiring revision | 5 |
| Maximum revision rounds | 2 |
| Total write-edit message exchanges | ~28 |
| Source files read by researcher | 8 |
| Final tutorial word count | ~9,500 |
| Total artifacts produced | 20 files (research notes + 9 drafts + 9 finals + 1 assembled) |
| Orchestrator interventions | 1 (go-ahead nudge to writer) |
| End-to-end time | ~45 minutes |
| Agents terminated cleanly | 4/4 (all confirmed shutdown_response) |

---

*Documentation written by the orchestrator (main Claude session) from direct observation of all agent activity.*
*Team: turboquant-tutorial-team | Platform: Claude Code with multi-agent framework*
