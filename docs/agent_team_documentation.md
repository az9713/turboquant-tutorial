# Agent Team Documentation
## Project: Munger Latticework Article — `munger-article-team`
### A Complete Technical Account of a Multi-Agent Document Production System

---

> **Purpose of this document:** To fully demystify how a team of AI agents collaborated to research, write, edit, and publish an 84-page PDF report on Charlie Munger's Latticework of Mental Models. Every agent, every turn, every message, every decision is documented here — nothing is left as a black box.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [The Agents — Full Profiles](#3-the-agents--full-profiles)
   - 3.1 The Orchestrator (Main Claude Session)
   - 3.2 The Researcher
   - 3.3 The Writer
   - 3.4 The Editor
   - 3.5 The PDF Builder (Ad-hoc Specialist)
4. [The Task System](#4-the-task-system)
5. [The Communication Protocol](#5-the-communication-protocol)
6. [Complete Turn-by-Turn Narrative](#6-complete-turn-by-turn-narrative)
   - Phase 0: Pre-Team Background Agent
   - Phase 1: Team Architecture
   - Phase 2: Research
   - Phase 3: The Write-Edit Loop (8 sections)
   - Phase 4: Task Completion Signalling
   - Phase 5: PDF Assembly
   - Phase 6: Shutdown
7. [All Artifacts Produced](#7-all-artifacts-produced)
8. [Inter-Agent Message Log](#8-inter-agent-message-log)
9. [Quality Metrics — The Write-Edit Loop](#9-quality-metrics--the-write-edit-loop)
10. [Key Design Principles Illustrated](#10-key-design-principles-illustrated)
11. [What Could Go Wrong — Edge Cases Encountered](#11-what-could-go-wrong--edge-cases-encountered)
12. [Summary Statistics](#12-summary-statistics)

---

## 1. Project Overview

### Goal
Produce an authoritative, beautifully formatted 84-page PDF report on Charlie Munger's Latticework of Mental Models, drawing from four source files (including Munger's actual 1994 USC speech) and extensive web research, written and edited to publication standard.

### The Team
A coordinated team of **3 named AI agents** plus a **human-facing orchestrator** (the main Claude session) and one **ad-hoc specialist** spawned mid-project.

### The Result
- **File:** `munger_latticework_mental_models.pdf`
- **Pages:** 84
- **Size:** 211 KB
- **Structure:** Part I (Reference Guide, 12 sections) + Part II (The Article, 8 polished prose sections)
- **Time:** ~45 minutes end-to-end

### Source Materials Read
| File | Lines | Content |
|---|---|---|
| `charlie_munger_usc_speech.txt` | ~550 | Munger's 1994 USC "Worldly Wisdom" speech — **primary new source** |
| `transcript.txt` | ~900 | Mohnish Pabrai SXSW talk on latticework |
| `gpt5_summary.txt` | ~1,900 | GPT-5 tutorial: 150 mental models across 6 disciplines |
| `gemini3_summary.txt` | ~100 | Gemini summary of Pabrai's models |

---

## 2. System Architecture

### Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR (Team Lead)                      │
│  • Designs team structure       • Monitors idle_notifications   │
│  • Creates tasks & dependencies  • Intervenes when stuck        │
│  • Spawns all agents            • Triggers shutdown             │
└───────────────────────────┬─────────────────────────────────────┘
                            │ spawns + manages
          ┌─────────────────┼─────────────────┐
          ▼                 ▼                 ▼
    ┌──────────┐      ┌──────────┐      ┌──────────┐
    │RESEARCHER│      │  WRITER  │      │  EDITOR  │
    │ Task #1  │      │ Task #2  │      │ Task #3  │
    └────┬─────┘      └────┬─────┘      └────┬─────┘
         │                 │  ◄────────────►  │
    research_notes.md   draft_section_N.md    │
         │              final_section_N.md    │
         └─────────────────┴──────────────────┘
                           │ all complete
                           ▼
                    ┌──────────────┐
                    │ PDF BUILDER  │
                    │  Task #4     │
                    │ (ad-hoc)     │
                    └──────┬───────┘
                           │
                    munger_latticework_
                    mental_models.pdf
                    (84 pages, 211 KB)
```

### Task Dependency Graph

```
Task #1: Research
    ├── blocks Task #2 (Write)
    └── blocks Task #3 (Edit)

Task #2: Write  ◄──► Task #3: Edit   (iterative loop)
    └── both block Task #4 (PDF Assembly)

Task #4: Assemble PDF
    └── terminal node — no downstream dependencies
```

### Agent Communication Topology

```
Orchestrator ──SendMessage──► Researcher
Orchestrator ──SendMessage──► Writer
Orchestrator ──SendMessage──► Editor
Researcher   ──SendMessage──► Writer     (handoff: research ready)
Writer       ──SendMessage──► Editor     (draft ready for critique)
Editor       ──SendMessage──► Writer     (critique / approval)
Editor       ──SendMessage──► Researcher (all sections approved → PDF)
```

Agents do **not** call each other's functions. All coordination is message-passing + shared filesystem.

---

## 3. The Agents — Full Profiles

---

### 3.1 The Orchestrator — Main Claude Session

| Property | Detail |
|---|---|
| **Identity** | The main Claude instance the user talks to directly |
| **Team role** | Team architect, task manager, traffic controller, emergency responder |
| **Agent type** | N/A — this is the root session, not a spawned agent |
| **Tools available** | All tools: `TeamCreate`, `TaskCreate`, `TaskUpdate`, `TaskList`, `Agent`, `SendMessage`, `Read`, `Write`, `Bash`, `WebSearch`, `Glob`, `Grep`, etc. |

#### What the Orchestrator Does

The orchestrator is the **brain of the operation** but is **not involved in content production**. It never writes a word of the article. Its job is entirely architectural and supervisory:

1. **Reads source files** before spawning the team — to understand scope, identify what's new, and write precise agent prompts
2. **Creates the team namespace** via `TeamCreate`
3. **Creates all 4 tasks** with descriptions, active forms, and dependency chains
4. **Spawns all 3 agents** simultaneously with detailed role prompts (~800–1,000 words each)
5. **Monitors `idle_notification` events** — passively receives agent state updates without polling
6. **Interprets notification summaries** — the `summary` field tells the orchestrator about peer-to-peer messages without exposing full content
7. **Intervenes selectively** — only when the pipeline stalls or an agent needs redirection
8. **Re-routes tasks** when the wrong agent is assigned (researcher → builder for PDF assembly)
9. **Issues shutdown requests** to all agents at the end

#### The Orchestrator's Three Interventions

**Intervention 1 — Closing Task #1 early:**
A background agent had already produced `research_notes.md` before the team was created. When the orchestrator received that agent's completion notification, it read the file, confirmed it was complete (681 lines), sent the researcher a "verify and hand off" message, manually marked Task #1 `completed`, and sent the writer the direct go-ahead. This prevented the pipeline from stalling while the team researcher re-did work already done.

**Intervention 2 — Sending the writer the go-ahead:**
After closing Task #1, the orchestrator sent the writer a detailed message including the 8-section structure, the top 5 USC speech findings, and specific guidance on which anecdotes to use in which sections. This gave the writer a concrete roadmap rather than leaving them to infer structure from the research notes alone.

**Intervention 3 — Re-routing Task #4:**
When the orchestrator assigned PDF assembly to the researcher, the researcher correctly pushed back ("please assign to whoever should handle PDF assembly"). The orchestrator recognised this as a capability mismatch — a research agent should not be debugging ReportLab Python — and spawned a specialist `builder` subagent instead.

#### Who the Orchestrator Talks To
- All agents via `SendMessage`
- The user directly (text output)
- Background agents via `Agent` tool spawn

---

### 3.2 The Researcher (`researcher@munger-article-team`)

| Property | Detail |
|---|---|
| **Name** | `researcher` |
| **Team ID** | `researcher@munger-article-team` |
| **Persona** | "Experienced professional researcher specialising in business, investing, and cognitive science" |
| **Agent type** | General-purpose (all tools) |
| **Assigned task** | Task #1 |
| **Task status at spawn** | Immediately claimable (no blockers) |

#### What the Researcher Does

The researcher is the **pipeline's first node** — nothing can move forward until it completes. Its entire job is information gathering and synthesis.

**Step 1 — Claims Task #1**
On waking, the researcher reads the team config at `~/.claude/teams/munger-article-team/config.json` to discover teammate names. Then calls `TaskList` to see available work. Finds Task #1 unblocked and unowned. Claims it via `TaskUpdate` (`owner=researcher`, `status=in_progress`).

**Step 2 — Reads all source files**
Source files are too large for single reads. The researcher uses `offset` and `limit` parameters to read in chunks:
- `charlie_munger_usc_speech.txt` — 3 chunks × 250 lines
- `transcript.txt` — 4 chunks × 250 lines
- `gpt5_summary.txt` — 7 chunks × 250 lines (largest file)
- `gemini3_summary.txt` — 1 chunk (short file)

**Step 3 — Conducts web research**
Six targeted `WebSearch` calls:
1. "Charlie Munger pari-mutuel racetrack stock market analogy"
2. "Charlie Munger fat pitch 20 punch card Warren Buffett selectivity"
3. "Charlie Munger circle of competence real examples Berkshire"
4. "Munger Buffett Washington Post See's Candy Coca-Cola mental models investment"
5. "Charlie Munger two track analysis rational psychological"
6. "Munger invert always invert Jacobi practical examples"

**Step 4 — Writes `research_notes.md`**
Uses the `Write` tool to produce a 681-line / 40 KB structured document with 9 parts:
- Part A: Core Philosophy (direct USC speech quotes)
- Part B: The Mental Disciplines (6 disciplines with Munger's exact words)
- Part C: Investing Application (pari-mutuel, fat-pitch, Berkshire case studies)
- Part D: Key Stories & Anecdotes (14 vivid stories, fully written out)
- Part E: Pabrai Extensions
- Part F: 150-Model Reference
- Part G: New vs. Existing PDF assessment
- Part H: Recommended 8-section article structure

**Step 5 — Notifies writer and closes task**
Sends `SendMessage` to `writer` with confirmation, top 5 USC findings, and section structure. Calls `TaskUpdate` to mark Task #1 `completed`.

#### Inputs
- 4 source text files (read via `Read`)
- 6 web search results (via `WebSearch`)

#### Output
- `research_notes.md` — 681 lines, 40 KB

#### Who the Researcher Talks To
| Direction | Recipient | Content |
|---|---|---|
| Receives | Orchestrator | Initial spawn prompt + "verify and hand off" message |
| Sends | Writer | "Research notes ready — here are the top 5 findings and structure" |
| Receives | Orchestrator | Task #4 assignment |
| Sends | Orchestrator | "Please assign Task #4 to whoever handles PDF assembly" |
| Receives | Orchestrator | Shutdown request |
| Sends | Orchestrator | Shutdown approved |

---

### 3.3 The Writer (`writer@munger-article-team`)

| Property | Detail |
|---|---|
| **Name** | `writer` |
| **Team ID** | `writer@munger-article-team` |
| **Persona** | "Experienced writer specialising in business and technical publications — The Economist, Harvard Business Review, Bloomberg Businessweek" |
| **Agent type** | General-purpose (all tools) |
| **Assigned task** | Task #2 |
| **Task status at spawn** | Blocked by Task #1 |

#### What the Writer Does

The writer is the **central production node** — the agent that actually produces the article. It works in a strict sequential loop, processing one section at a time and never advancing until the previous section is editor-approved.

**The Per-Section Loop (executed 8 times):**

```
For each section N (1 through 8):

  1. Read research_notes.md (Part relevant to section N)
  2. Draft section prose (600–900 words)
  3. Write to draft_section_N.md via Write tool
  4. SendMessage to editor: "Section N draft ready — please read draft_section_N.md"
  5. Go idle (await editor response)

  [Editor sends critique via SendMessage]

  6. Wake from idle, read critique
  7. Revise draft addressing all specific points
  8. Write to final_section_N.md via Write tool
  9. SendMessage to editor: "Revised Section N done — ready for review"
  10. Go idle (await approval)

  [Editor sends approval via SendMessage]

  11. Wake from idle, receive "Section N approved — proceed to Section N+1"
  12. Repeat for Section N+1
```

**The 8 Sections Written:**

| N | Title | Core Content |
|---|---|---|
| 1 | The Man Who Read Everything | Munger biography, intellectual restlessness, the reading habit |
| 2 | The Latticework — What It Is and Why It Matters | USC speech core: isolated facts fail, models must interconnect, hammer/nail problem |
| 3 | The Mental Disciplines — Building the Toolkit | All 6 disciplines with Munger's exact words and vivid examples |
| 4 | The Psychology of Human Misjudgment | 25 biases, two-track analysis, magician analogy, Federal Express story |
| 5 | The Investing Mind | Pari-mutuel analogy, fat-pitch, 20-punch-card, Washington Post, See's Candy |
| 6 | The Lollapalooza Effect | Cascading forces, Tupperware, bubbles, Costco, Walmart palookas |
| 7 | Pabrai's Extensions | Truth on log scale, asymmetric bets, Slide 10 story, 168-hour week |
| 8 | Building Your Own Latticework | 7-step guide, priority 100 models, checklist method, reading list |

#### Writing Standards Applied

The writer was prompted to apply these standards to every section:
- **Business-publication quality:** clear, direct, no filler
- **Every abstract claim backed by a concrete example** — named companies, real situations
- **Vivid and memorable** — reader should be able to teach these ideas after reading
- **Direct Munger quotes as anchor points** — pulled from research_notes.md
- **Strong opening sentence, clear logical spine, punchy close**
- **No listicles** — flowing prose with examples woven in

#### Inputs
- `research_notes.md` (via `Read`)
- Go-ahead message from orchestrator (via `SendMessage`)
- Editor critiques (via `SendMessage`) — one or two per section

#### Outputs
- `draft_section_1.md` through `draft_section_8.md` (first drafts)
- `final_section_1.md` through `final_section_8.md` (revised finals)

#### Who the Writer Talks To
| Direction | Recipient | Content |
|---|---|---|
| Receives | Orchestrator | "Stand by" (early) + go-ahead with 8-section structure |
| Receives | Researcher | "Research notes ready at [path]" |
| Sends | Editor | "Section N draft ready — please read draft_section_N.md" (×8) |
| Receives | Editor | Critiques (×8) |
| Sends | Editor | "Revised Section N done" (×8) |
| Receives | Editor | "Section N approved — proceed to N+1" (×8) |
| Sends | Editor | "All 8 sections finalized and approved" |
| Receives | Orchestrator | Shutdown request |
| Sends | Orchestrator | "All writing work complete. Shutting down." + shutdown approved |

---

### 3.4 The Editor (`editor@munger-article-team`)

| Property | Detail |
|---|---|
| **Name** | `editor` |
| **Team ID** | `editor@munger-article-team` |
| **Persona** | "Demanding, exacting editor for a top-tier business and intellectual publication — The Economist meets Harvard Business Review" |
| **Agent type** | General-purpose (all tools) |
| **Assigned task** | Task #3 |
| **Task status at spawn** | Blocked by Task #1 |

#### What the Editor Does

The editor is the **quality control node** — the gatekeeper that ensures nothing substandard reaches the final PDF. Every section must pass through the editor before the writer can advance.

**Preparation Phase:**
When unblocked by Task #1 completion, the editor proactively reads `research_notes.md` in full before receiving any drafts. This allows it to:
- Calibrate its accuracy checks (know what Munger actually said)
- Know which anecdotes and examples should appear in which sections
- Identify missing material that the writer overlooked

**The Per-Section Critique Loop:**

```
For each section N:

  1. Receive SendMessage from writer: "Section N draft ready"
  2. Read draft_section_N.md via Read tool
  3. Cross-check against research_notes.md for accuracy and completeness
  4. Apply 7-criterion evaluation framework (see below)
  5. Write critique — specific, actionable, pointing to exact sentences
  6. SendMessage to writer with full critique
  7. Go idle (await revision)

  [Writer sends revision]

  8. Read final_section_N.md
  9. Assess whether critique was addressed
  10. If yes: SendMessage "Section N approved — proceed to Section N+1"
  11. If no: send second-round critique (never happened — all approved in round 2)
  12. Go idle
```

#### The 7-Criterion Evaluation Framework

Every section was evaluated on all 7 criteria:

**1. The Opening Hook**
- Is the first sentence grabbing? Does it start with a scene, a paradox, a startling fact?
- Rejected: abstract statements, biographical summaries, "In this section we will..."
- Required: a concrete moment, a vivid detail, an unexpected angle

**2. Accuracy to Munger's Ideas**
- Are quotes accurate? Checked word-for-word against research_notes.md
- Are ideas faithful to what Munger actually argued?
- Has anything been oversimplified to the point of distortion?

**3. Depth vs. Superficiality**
- Are examples specific (named companies, named people, named events)?
- Does the section teach something the reader can USE, or just describe?
- Rejected: "For example, many companies have used this model successfully."
- Required: "When Berkshire bought The Washington Post in 1973 at one-fifth of its private market value..."

**4. Logical Flow**
- Does the section have a single organising argument (a spine)?
- Does each paragraph follow from the previous?
- Are there any non-sequiturs or sudden jumps?

**5. The Key Stories**
- Are the best anecdotes from the research notes present?
- Are they developed with enough detail to be memorable?
- Or just name-dropped without development?

**6. The Closing**
- Does the section end with a punch — a memorable phrase, a callback, a forward hook?
- Or does it trail off?

**7. Missing Material**
- What important content from research_notes.md should have been included but wasn't?
- What connections to themes from other sections are being missed?

#### Critique Style
The editor was explicitly instructed to be **specific and pointed**, not vague. The difference:
- ❌ "The third paragraph could be improved."
- ✅ "The third paragraph opens with a cliché ('In today's complex world...'). Replace it with the Federal Express pay-per-shift story from Part D of the research notes — that's the perfect concrete illustration of incentive power."

#### Inputs
- `research_notes.md` (read proactively at start)
- `draft_section_N.md` (×8, received via notification)
- `final_section_N.md` (×8, read after writer's revisions)

#### Outputs
- Critique messages to writer (×8 via `SendMessage`)
- Approval messages to writer (×8 via `SendMessage`)
- Final "all sections approved" message to researcher

#### Who the Editor Talks To
| Direction | Recipient | Content |
|---|---|---|
| Receives | Orchestrator | "Read research notes — prepare for first draft" |
| Receives | Writer | "Section N draft ready" (×8) |
| Sends | Writer | Full critiques (×8) |
| Sends | Writer | "Section N approved — proceed to N+1" (×8) |
| Sends | Researcher | "All 8 sections approved — ready for PDF assembly" |
| Receives | Orchestrator | Shutdown request |
| Sends | Orchestrator | Shutdown approved |

---

### 3.5 The PDF Builder (Ad-hoc Specialist)

| Property | Detail |
|---|---|
| **Name** | Not a named team member — one-off `builder` subagent |
| **Spawned by** | Orchestrator (directly, after researcher declined Task #4) |
| **Agent type** | `builder` (specialised for code writing and file manipulation) |
| **Assigned task** | Task #4 (PDF assembly) |
| **Communication** | Does not use `SendMessage` — reports only via task result to orchestrator |

#### Why It Exists Separately

The researcher was correctly specialised for reading, synthesising, and writing prose. When assigned PDF assembly (which required reading and modifying 2,600 lines of ReportLab Python), it correctly pushed back. The orchestrator spawned this specialist instead — a `builder` agent with file-editing and Bash execution capabilities.

#### What the Builder Does

1. Reads all 8 `final_section_N.md` files
2. Reads `research_notes.md` for full context
3. Reads existing `generate_munger_pdf.py` (2,600 lines) to understand current structure and style
4. Adds helper functions:
   - `_art_quote()` — styled Munger quote boxes (light blue background, gold border)
   - `_art_callout()` — vivid story callout boxes (green-tinted, gold border)
   - `_part2_divider()` — full-page navy divider with "PART II / Munger In His Own Words"
   - `_md_to_flowables_rich()` — intelligent markdown parser that detects blockquotes, named stories, headers
5. Adds Part II chapter (8 sections, p.52–84)
6. Updates Table of Contents with Part I and Part II entries
7. Runs `python3 generate_munger_pdf.py` via `Bash`
8. Confirms output: 84 pages, 211 KB

#### Inputs
- `final_section_1.md` through `final_section_8.md`
- `research_notes.md`
- `generate_munger_pdf.py` (existing script)

#### Output
- Updated `generate_munger_pdf.py`
- `munger_latticework_mental_models.pdf` — 84 pages, 211 KB

---

## 4. The Task System

### What Tasks Are

Tasks are **shared state objects** stored in `~/.claude/tasks/munger-article-team/`. Every agent on the team can read and update them. They serve three purposes simultaneously:
1. **Work queue** — agents check for unowned, unblocked tasks to claim
2. **Progress tracker** — `status` field (`pending` / `in_progress` / `completed`) shows pipeline state
3. **Dependency enforcer** — `blockedBy` prevents agents from starting work they shouldn't do yet

### Task Definitions

**Task #1 — Research**
```
Subject:     Read all source files and do deep web research on Munger's latticework
Status flow: pending → in_progress (researcher) → completed (orchestrator)
Blockers:    None
Blocks:      Tasks #2 and #3
Owner:       researcher
```

**Task #2 — Write Article**
```
Subject:     Draft all 8 sections of the Munger latticework article
Status flow: pending [blocked] → in_progress (writer) → completed (writer)
Blockers:    Task #1
Blocks:      Task #4
Owner:       writer
```

**Task #3 — Edit & Critique**
```
Subject:     Review each section draft and send critique to writer
Status flow: pending [blocked] → in_progress (editor) → completed (editor)
Blockers:    Task #1
Blocks:      Task #4
Owner:       editor
```

**Task #4 — Assemble PDF**
```
Subject:     Integrate all finalized sections into the updated PDF report
Status flow: pending [blocked] → in_progress (researcher → builder) → completed (orchestrator)
Blockers:    Tasks #2 AND #3
Blocks:      Nothing (terminal)
Owner:       researcher (assigned), builder (actually executed)
```

### The Dependency System in Practice

When Task #1 was `in_progress`, `TaskList` showed:
```
#2 [pending] Write Article [blocked by #1]
#3 [pending] Edit & Critique [blocked by #1]
```

Agents saw this and understood there was nothing to do. No message from the orchestrator was needed to keep them idle — the task system handled it.

When Task #1 was marked `completed`, Tasks #2 and #3 became `pending` with no blockers — a clear signal to both writer and editor that they could begin. The orchestrator then also sent an explicit go-ahead message to the writer, but the task system had already structurally enabled it.

### Who Updated Which Tasks

| Task | Event | Updated by | Method |
|---|---|---|---|
| #1 | Claimed | Researcher | `TaskUpdate(owner=researcher, status=in_progress)` |
| #1 | Completed | Orchestrator | `TaskUpdate(status=completed)` — manually, after background agent finished |
| #2 | Claimed | Writer | `TaskUpdate(owner=writer, status=in_progress)` |
| #2 | Completed | Writer | `TaskUpdate(status=completed)` — self-reported after all 8 sections done |
| #3 | Claimed | Editor | `TaskUpdate(owner=editor, status=in_progress)` |
| #3 | Completed | Editor | `TaskUpdate(status=completed)` — self-reported after all 8 approvals |
| #4 | Assigned | Orchestrator | `TaskUpdate(owner=researcher, status=in_progress)` |
| #4 | Completed | Orchestrator | `TaskUpdate(status=completed)` — after builder confirmed PDF done |

---

## 5. The Communication Protocol

### The `SendMessage` Tool

`SendMessage` is the **only channel** through which agents communicate with each other. Key properties:
- Addressed by **name** (`to: "writer"`, `to: "editor"`) — never by ID
- Includes a `summary` field (5–10 words) shown in the orchestrator's UI as a preview
- Delivery is **asynchronous** — the sender goes idle; the recipient wakes when the message arrives
- Messages from teammates are **automatically delivered** — the orchestrator does not need to check an inbox
- Agents cannot read each other's files directly unless they know the filename — the message must tell them where to look

### The `idle_notification` Signal

Every time an agent completes a processing turn, the system sends an `idle_notification` to the orchestrator automatically. This is how the orchestrator tracked pipeline health without polling.

The notification has two forms:

**Simple idle (agent waiting, nothing to report):**
```json
{"type": "idle_notification", "from": "writer", "timestamp": "...", "idleReason": "available"}
```

**Idle with peer message summary (agent sent a message to another agent):**
```json
{
  "type": "idle_notification",
  "from": "editor",
  "idleReason": "available",
  "summary": "[to writer] Section 3 approved — proceed to Section 4"
}
```

The `summary` field gave the orchestrator **visibility into peer-to-peer communication** without seeing the full message content. This is how the orchestrator knew Section 3 was approved without the editor explicitly reporting to it.

### Idle Notification Interpretation Guide

| Who | Summary / State | Meaning | Orchestrator action |
|---|---|---|---|
| writer | idle, no summary | Waiting for something | Checked task status |
| editor | idle, no summary | Waiting for draft | No action needed |
| researcher | idle, `[to writer] Research notes complete` | Handoff sent | Closed Task #1, sent writer go-ahead |
| writer | idle, `[to editor] Section N draft ready` | Draft sent, awaiting critique | Noted progress, no action |
| editor | idle, `[to writer] Section N critique — strong draft` | Critique delivered | No action — loop running |
| editor | idle, `[to writer] Section N approved` | Section done | Noted N/8 progress |
| writer | idle, `[to editor] All 8 sections finalized` | Writing complete | Triggered Task #4 |
| researcher | idle, `PDF assembled successfully` | PDF done | Closed Task #4, began shutdown |

### The Shutdown Protocol

At the end of the project, the orchestrator used a formal handshake to terminate agents:

**Orchestrator sends:**
```json
{"type": "shutdown_request", "reason": "Project complete."}
```

**Agent acknowledges and terminates:**
```json
{"type": "shutdown_approved", "requestId": "shutdown-1774497074709@writer"}
```

**System confirms:**
```
{"type": "teammate_terminated", "message": "writer has shut down."}
```

The researcher required two shutdown requests — it was mid-task (idle with `interrupted` status) after its PDF assembly attempt when the first request arrived. A second request was sent and it then confirmed and terminated.

### File-Based Handoffs

In addition to messages, agents coordinated through the filesystem:

| From | To | File | Protocol |
|---|---|---|---|
| Researcher | Writer | `research_notes.md` | Researcher writes file, then sends message telling writer the path |
| Writer | Editor | `draft_section_N.md` | Writer writes file, sends message telling editor the filename |
| Writer | Builder | `final_section_N.md` | Writer writes files; builder reads them after orchestrator spawns it |
| Builder | Orchestrator | `munger_latticework_mental_models.pdf` | Builder writes file, reports path + stats in task result |

---

## 6. Complete Turn-by-Turn Narrative

### Phase 0: Pre-Team Background Agent

**What happened:**
Before `TeamCreate` was called, the user asked to spawn agents. The orchestrator launched a single background research agent (`a64cdbc382bd45db1`) using the `Agent` tool directly, without team membership.

The user interrupted and asked to use `TeamCreate` instead. The orchestrator switched approach, but the background agent was already running and could not be stopped.

**The background agent's behaviour:**
- Read all 4 source files in chunks
- Conducted 6 web searches
- Wrote `research_notes.md` (681 lines / 40 KB)
- Tried to notify "munger-writer" — but there was no such agent yet; it attempted a Slack message instead (failed)
- Completed after ~10 minutes and sent a task-completion notification to the orchestrator

**Why this mattered:**
The background agent had no team membership, no task awareness, and no ability to use `SendMessage` to team members (they didn't exist yet). But it *did* produce the key output file — `research_notes.md`. The orchestrator later used this file to accelerate the team's pipeline.

**Lesson:** Background agents and team agents can coexist but don't share context. The orchestrator must bridge any handoffs manually.

---

### Phase 1: Team Architecture

**Turn 1 — Tool loading**
The orchestrator called `ToolSearch` to load `TeamCreate`, `TaskCreate`, `TaskUpdate`, `TaskList`, and `SendMessage` tools (they were in deferred state and needed loading before use).

**Turn 2 — Team creation**
```
TeamCreate(
  team_name="munger-article-team",
  description="Three-agent team to research, write, and edit..."
)
```
This created:
- Team config at `~/.claude/teams/munger-article-team/config.json`
- Shared task list namespace at `~/.claude/tasks/munger-article-team/`

**Turn 3 — Task creation**
Four `TaskCreate` calls, then three `TaskUpdate` calls to set dependencies:
```
TaskUpdate(taskId=2, addBlockedBy=["1"])
TaskUpdate(taskId=3, addBlockedBy=["1"])
TaskUpdate(taskId=4, addBlockedBy=["2", "3"])
```

**Turn 4 — Agent spawning**
Three `Agent` tool calls in a single message (all run in background simultaneously):
```
Agent(name="researcher", team_name="munger-article-team", run_in_background=true, prompt=...)
Agent(name="writer",     team_name="munger-article-team", run_in_background=true, prompt=...)
Agent(name="editor",     team_name="munger-article-team", run_in_background=true, prompt=...)
```

All three agents woke up, read the team config, and checked `TaskList`:
- Researcher: found Task #1 unblocked → claimed it
- Writer: found Task #2 blocked → sent idle notification to orchestrator
- Editor: found Task #3 blocked → sent idle notification to orchestrator

---

### Phase 2: Research

**Researcher Turn 1:** Claimed Task #1. Began reading source files.

**Researcher Turns 2–15 (approximately):**
14+ `Read` tool calls (with offset/limit for large files) + 6 `WebSearch` calls.

Key reads:
- `charlie_munger_usc_speech.txt`: read as 3 chunks (lines 1–250, 251–500, 500–end)
- `gpt5_summary.txt`: read as 7 chunks (the largest file at 1,900+ lines)

Key searches:
- Pari-mutuel analogy, fat-pitch strategy, circle of competence examples, Berkshire case studies, inversion/Jacobi, worldly wisdom advantage

**Researcher Turn 16:** `Write` tool call — produced `research_notes.md` (681 lines).

**Researcher Turn 17:** `SendMessage` to `writer` + `TaskUpdate(status=completed)`.

**Concurrent — Orchestrator receives background agent completion:**
`task-notification` arrived confirming `a64cdbc382bd45db1` completed. Orchestrator:
1. Called `Read` on `research_notes.md` to verify (confirmed 681 lines, well-structured)
2. Called `SendMessage` to `researcher`: "File already exists — verify and hand off"
3. Called `SendMessage` to `editor`: "Read research notes now — prepare"
4. Called `TaskUpdate(taskId=1, status=completed)` — manually closed Task #1
5. Called `SendMessage` to `writer`: detailed go-ahead with 8-section structure

This unblocked Tasks #2 and #3 explicitly.

---

### Phase 3: The Write-Edit Loop

The loop ran 8 times. Here is the complete record:

#### Section 1: The Man Who Read Everything

| Sub-turn | Agent | Action | Output |
|---|---|---|---|
| W1 | Writer | Read research_notes.md (Part A + Part G) | — |
| W2 | Writer | Write draft_section_1.md | 5.3 KB |
| W3 | Writer | SendMessage to editor: "Section 1 draft ready" | — |
| E1 | Editor | Read draft_section_1.md | — |
| E2 | Editor | Cross-check vs research_notes.md | — |
| E3 | Editor | SendMessage to writer: critique | "Strong draft, specific fixes needed" |
| W4 | Writer | Read critique, revise draft | — |
| W5 | Writer | Write final_section_1.md | 6.0 KB |
| W6 | Writer | SendMessage to editor: "Revised Section 1 done" | — |
| E4 | Editor | Read final_section_1.md | — |
| E5 | Editor | SendMessage to writer: "Section 1 approved — proceed to Section 2" | ✅ |

**Time for this section:** ~4 minutes

#### Section 2: The Latticework — What It Is

| Sub-turn | Agent | Action | File size |
|---|---|---|---|
| Write draft | Writer | draft_section_2.md | 7.1 KB |
| Critique | Editor | "Strong core, five specific fixes" | — |
| Revise | Writer | final_section_2.md | 7.6 KB |
| Approve | Editor | ✅ | — |

#### Section 3: The Mental Disciplines

| Sub-turn | Agent | Action | File size |
|---|---|---|---|
| Write draft | Writer | draft_section_3.md | 11 KB |
| Critique | Editor | "Best draft yet, four fixes" | — |
| Revise | Writer | final_section_3.md | 12 KB |
| Approve | Editor | ✅ | — |

**Notable:** Editor called this the "best draft yet" — indicating the writer was learning the editor's standards and pre-applying them.

#### Section 4: The Psychology of Human Misjudgment

| Sub-turn | Agent | Action | File size |
|---|---|---|---|
| Write draft | Writer | draft_section_4.md | 9.1 KB |
| Critique | Editor | "Strongest draft, three fixes" | — |
| Revise | Writer | final_section_4.md | 11 KB |
| Approve | Editor | ✅ | — |

**Notable:** Largest relative size increase (9.1 → 11 KB). The editor pushed for more content: likely asked for the Federal Express story to be told in full and the two-track analysis explained with examples.

#### Section 5: The Investing Mind

| Sub-turn | Agent | Action | File size |
|---|---|---|---|
| Write draft | Writer | draft_section_5.md | 10 KB |
| Critique | Editor | "Near-final, two fixes" | — |
| Revise | Writer | final_section_5.md | 9.9 KB |
| Approve | Editor | ✅ | — |

**Notable:** The only section where the final was *smaller* than the draft (10 → 9.9 KB). The editor's two fixes were likely about prose tightening, not adding content.

#### Section 6: The Lollapalooza Effect

| Sub-turn | Agent | Action | File size |
|---|---|---|---|
| Write draft | Writer | draft_section_6.md | 8.4 KB |
| Critique | Editor | "Strong, three fixes" | — |
| Revise | Writer | final_section_6.md | 9.0 KB |
| Approve | Editor | ✅ | — |

#### Section 7: Pabrai's Extensions

| Sub-turn | Agent | Action | File size |
|---|---|---|---|
| Write draft | Writer | draft_section_7.md | 9.4 KB |
| Critique | Editor | "Good, three fixes needed" | — |
| Revise | Writer | final_section_7.md | 11 KB |
| Approve | Editor | ✅ | — |

**Notable:** "Good" rather than "strong" — the editor may have found this section relied too heavily on bullet-point style rather than prose narrative (Pabrai's models are naturally list-like). The large revision (+1.6 KB) suggests the writer converted lists into flowing prose.

#### Section 8: Building Your Own Latticework

| Sub-turn | Agent | Action | File size |
|---|---|---|---|
| Write draft | Writer | draft_section_8.md | 9.5 KB |
| Critique | Editor | "Near-final, two fixes" | — |
| Revise | Writer | final_section_8.md | 11 KB |
| Approve | Editor | ✅ |  |

**After Section 8 approval:**
- Writer sent two messages to editor: "Revised Section 8 done" and "All 8 sections finalized and approved"
- Writer called `TaskUpdate(status=completed)` on Task #2
- Editor called `TaskUpdate(status=completed)` on Task #3
- Editor sent message to researcher: "All 8 sections approved — ready for PDF assembly"

---

### Phase 4: Task Completion and PDF Assembly Trigger

**Orchestrator received `idle_notification` from writer** with summary: "[to editor] All 8 sections finalized and approved"

**Orchestrator checked `TaskList`:**
```
#1 [completed]
#2 [completed]
#3 [completed]
#4 [pending] — now unblocked
```

**Orchestrator assigned Task #4:**
```
TaskUpdate(taskId=4, status=in_progress, owner=researcher)
SendMessage(to=researcher, message="All 8 sections done — assemble PDF now...")
```

**Researcher response:**
Received the message, assessed the task, recognised it required Python/ReportLab expertise, sent message to orchestrator: "Task #4 is unblocked — please assign to whoever should handle PDF assembly."

**Orchestrator spawned builder subagent:**
```
Agent(
  subagent_type="builder",
  prompt="Read all 8 final sections, read generate_munger_pdf.py, add Part II chapter, run script..."
)
```

---

### Phase 5: PDF Assembly (Builder Agent)

The builder agent ran ~20 turns:

**Turns 1–10:** Read all 8 `final_section_N.md` files, read `research_notes.md`, read `generate_munger_pdf.py` (2,600 lines).

**Turns 11–16:** Wrote new helper functions into the script:
- `_art_quote()` — styled quote boxes
- `_art_callout()` — vivid story callout boxes
- `_part2_divider()` — Part II title page
- `_md_to_flowables_rich()` — intelligent markdown parser

**Turn 17:** Added Part II chapter builder to the main document assembly function.

**Turn 18:** Updated Table of Contents to show Part I and Part II entries.

**Turn 19:** Executed `python3 generate_munger_pdf.py` via `Bash`.

**Turn 20:** Confirmed output — 84 pages, 211 KB. Reported to orchestrator.

**Concurrent (slight duplication):** The researcher, after receiving the orchestrator's Task #4 assignment, also attempted a PDF assembly and produced an 82-page / 208 KB version. The builder's version (84 pages / 211 KB) ran later and is the file on disk.

---

### Phase 6: Shutdown

**Orchestrator → All agents:**
```
SendMessage(to="writer",     message={"type": "shutdown_request", "reason": "Project complete."})
SendMessage(to="editor",     message={"type": "shutdown_request", "reason": "Project complete."})
SendMessage(to="researcher", message={"type": "shutdown_request", "reason": "Project complete."})
```

**Writer:** Acknowledged with plain text "All writing work complete. Shutting down." System emitted `teammate_terminated: writer`.

**Editor:** Confirmed `shutdown_approved`. System emitted `teammate_terminated: editor`.

**Researcher:** First went `interrupted` (mid-task idle). Orchestrator sent a second `shutdown_request`. Researcher then confirmed `shutdown_approved`. System emitted `teammate_terminated: researcher`.

---

## 7. All Artifacts Produced

### Intermediate Files (Working Documents)

| File | Size | Producer | Purpose |
|---|---|---|---|
| `research_notes.md` | 40 KB / 681 lines | Researcher (+ background agent) | Foundation for all article writing |
| `draft_section_1.md` | 5.3 KB | Writer | First draft — biography |
| `draft_section_2.md` | 7.1 KB | Writer | First draft — latticework concept |
| `draft_section_3.md` | 11 KB | Writer | First draft — mental disciplines |
| `draft_section_4.md` | 9.1 KB | Writer | First draft — psychology |
| `draft_section_5.md` | 10 KB | Writer | First draft — investing mind |
| `draft_section_6.md` | 8.4 KB | Writer | First draft — lollapalooza effect |
| `draft_section_7.md` | 9.4 KB | Writer | First draft — Pabrai extensions |
| `draft_section_8.md` | 9.5 KB | Writer | First draft — practical guide |
| `final_section_1.md` | 6.0 KB | Writer (post-critique) | Approved section 1 |
| `final_section_2.md` | 7.6 KB | Writer (post-critique) | Approved section 2 |
| `final_section_3.md` | 12 KB | Writer (post-critique) | Approved section 3 |
| `final_section_4.md` | 11 KB | Writer (post-critique) | Approved section 4 |
| `final_section_5.md` | 9.9 KB | Writer (post-critique) | Approved section 5 |
| `final_section_6.md` | 9.0 KB | Writer (post-critique) | Approved section 6 |
| `final_section_7.md` | 11 KB | Writer (post-critique) | Approved section 7 |
| `final_section_8.md` | 11 KB | Writer (post-critique) | Approved section 8 |

### Final Deliverables

| File | Size | Pages | Producer |
|---|---|---|---|
| `generate_munger_pdf.py` | ~2,800 lines | — | Builder (updated existing script) |
| `munger_latticework_mental_models.pdf` | 211 KB | 84 | Builder |

### Total intermediate content produced: ~105 KB of markdown across 17 files

---

## 8. Inter-Agent Message Log

A complete record of every `SendMessage` call made during the project:

| # | From | To | Summary | Trigger |
|---|---|---|---|---|
| 1 | Orchestrator | Writer | "Stand by — researcher still working" | Writer went idle early |
| 2 | Orchestrator | Researcher | "Research notes exist — verify and hand off" | Background agent completed |
| 3 | Orchestrator | Editor | "Read research notes — prepare for first draft" | Research complete |
| 4 | Orchestrator | Writer | "Research complete — begin Section 1 now" (with 8-section structure) | Task #1 closed |
| 5 | Researcher | Writer | "Research notes complete and ready at [path]" | Task #1 completion |
| 6 | Writer | Editor | "Section 1 draft ready — please read draft_section_1.md" | Draft written |
| 7 | Editor | Writer | Section 1 critique: "Strong draft, specific fixes needed" | Draft read |
| 8 | Writer | Editor | "Revised Section 1 done — ready for review" | Revision written |
| 9 | Editor | Writer | "Section 1 approved — proceed to Section 2" | Revision read |
| 10–25 | Writer/Editor | Editor/Writer | Sections 2–8 draft / critique / revise / approve cycle (×7 more) | Loop |
| 26 | Writer | Editor | "All 8 sections finalized and approved" | Section 8 revision sent |
| 27 | Editor | Researcher | "All 8 sections approved — ready for PDF assembly" | All sections done |
| 28 | Orchestrator | Researcher | "All 8 sections done — assemble PDF now" | Tasks #2 + #3 completed |
| 29 | Researcher | Orchestrator | "Please assign Task #4 to whoever handles PDF" | Task #4 received |
| 30 | Orchestrator | Researcher | "PDF complete — shut down" | Builder finished |
| 31 | Orchestrator | Writer | "Project complete — shut down" | PDF confirmed |
| 32 | Orchestrator | Editor | "Project complete — shut down" | PDF confirmed |
| 33 | Orchestrator | Researcher | `shutdown_request` (formal JSON) | Shutdown phase |
| 34 | Orchestrator | Writer | `shutdown_request` (formal JSON) | Shutdown phase |
| 35 | Orchestrator | Editor | `shutdown_request` (formal JSON) | Shutdown phase |
| 36 | Writer | Orchestrator | "All writing complete. Shutting down." + `shutdown_approved` | Received shutdown_request |
| 37 | Editor | Orchestrator | `shutdown_approved` | Received shutdown_request |
| 38 | Orchestrator | Researcher | Second `shutdown_request` | Researcher went interrupted |
| 39 | Researcher | Orchestrator | `shutdown_approved` | Received second shutdown_request |

**Total SendMessage calls: ~39**
**Peer-to-peer messages (writer ↔ editor): ~24 (3 per section × 8 sections)**

---

## 9. Quality Metrics — The Write-Edit Loop

### Size Delta: Draft vs. Final

| Section | Draft | Final | Delta | Interpretation |
|---|---|---|---|---|
| 1 | 5.3 KB | 6.0 KB | +0.7 KB (+13%) | Added material: likely more vivid Munger biography details |
| 2 | 7.1 KB | 7.6 KB | +0.5 KB (+7%) | Minor additions: probably more USC speech quotes |
| 3 | 11 KB | 12 KB | +1.0 KB (+9%) | Added examples to disciplines — editor asked for specificity |
| 4 | 9.1 KB | 11 KB | +1.9 KB (+21%) | Largest gain: Federal Express story fully developed, two-track analysis expanded |
| 5 | 10 KB | 9.9 KB | -0.1 KB (-1%) | Tightened prose: editor asked for concision, not more content |
| 6 | 8.4 KB | 9.0 KB | +0.6 KB (+7%) | Added story development: Walmart palookas or Costco detail |
| 7 | 9.4 KB | 11 KB | +1.6 KB (+17%) | Lists converted to prose: Pabrai models told as narrative |
| 8 | 9.5 KB | 11 KB | +1.5 KB (+16%) | Priority 100 models expanded, reading list annotated |

### Editor Quality Progression

| Section | Editor Round 1 Verdict | Fixes Requested | Quality Signal |
|---|---|---|---|
| 1 | "Strong draft, specific fixes needed" | Unspecified number | Baseline quality |
| 2 | "Strong core, five specific fixes" | 5 | Good but needs work |
| 3 | "Best draft yet, four fixes" | 4 | Quality rising |
| 4 | "Strongest draft, three fixes" | 3 | Clear upward trend |
| 5 | "Near-final, two fixes" | 2 | Writer pre-applying editor standards |
| 6 | "Strong, three fixes" | 3 | Slight regression (new section type) |
| 7 | "Good, three fixes needed" | 3 | Style challenge (list-heavy material) |
| 8 | "Near-final, two fixes" | 2 | Returns to high quality for finale |

**Key insight:** The quality arc shows the writer genuinely internalising the editor's standards over the course of the project. Sections 3–5 showed consistent improvement. Sections 6–7 showed a slight regression when the material type changed (the Lollapalooza and Pabrai sections are more example-dense and list-prone). Section 8 recovered to near-final quality.

**All 8 sections were approved in exactly 2 rounds — no section required a third round.**

---

## 10. Key Design Principles Illustrated

### 10.1 Agents Communicate Only Through Messages and Files — Never Directly

No agent calls another agent's functions or reads another agent's internal state. The writer cannot compel the editor to do anything — it can only write a file and send a message saying "please read this file and send me feedback." The editor cannot edit the writer's draft — it can only send critique and let the writer decide what to change. This **loose coupling** is fundamental to multi-agent systems: it makes each agent independently testable, replaceable, and understandable.

### 10.2 The Task List Is Shared Ground Truth

Every agent had access to the same `TaskList`. Dependencies (`blockedBy`) were structural constraints, not conventions. This meant:
- The orchestrator did not need to tell each agent "wait for X" — the task system enforced it
- Any agent could check overall project state at any time
- Task status changes (completions) automatically unlocked downstream work

### 10.3 `idle_notification` Is the Heartbeat — Not a Problem Signal

New practitioners often misread idle notifications as errors. In this system, idle was the **normal resting state** between turns. Every agent went idle after each processing burst. The summary field on peer-message idles gave the orchestrator passive visibility into the pipeline without requiring active monitoring or polling.

### 10.4 Specialisation Matters — Capability Must Match Task

The researcher was correctly specialised for reading, synthesising, and writing prose. When given a Python/ReportLab task, it correctly declined. The editor was correctly specialised for critique — it never tried to rewrite the draft itself, only to guide the writer. The builder was correctly specialised for code and file manipulation. **Mismatching capability to task produces bad outcomes or correct refusals.**

### 10.5 The Orchestrator Intervenes Selectively — Not Constantly

The orchestrator monitored 40+ idle notifications during the project. It acted on 3 of them (Task #1 closure, go-ahead message, Task #4 re-routing). The other 37+ were acknowledged and dismissed. **Over-intervention would have broken the loop** — if the orchestrator had tried to manage every writer-editor exchange, it would have created bottlenecks and confusion. The skill is knowing which events require action and which are just the system breathing.

### 10.6 The Feedback Loop Improves Quality Over Time

16 editorial cycles across 8 sections produced measurable quality improvement — from "strong draft, specific fixes" to "near-final, two fixes." This is emergent: neither the writer nor the editor was explicitly told "your quality is improving." The improvement came from the writer receiving concrete, specific feedback and applying it to the next section before submitting. **Structured feedback loops produce compounding quality gains.**

### 10.7 Background Agents and Team Agents Are Different Systems

The background research agent (`a64cdbc382bd45db1`) had no team ID, no access to `SendMessage` for team members, no task awareness, and no ability to update team tasks. It was a standalone subprocess. Its only output was a file and a completion notification to the orchestrator. The orchestrator had to manually bridge the gap — this is a critical architectural distinction that practitioners must understand.

### 10.8 Parallel Spawning Is More Efficient Than Sequential

All 3 agents were spawned in a single message. This meant all three were alive and ready from the moment the team was created. The writer and editor were correctly idle (blocked by Task #1) rather than not-yet-existing. When Task #1 completed, they were **immediately available** to begin work — no spawn latency at that point. Always spawn team members upfront, even if they'll be idle initially.

### 10.9 File Paths Are First-Class Communication

Half the coordination in this system was file-path-based, not message-based. The researcher told the writer: "The research notes are at `<project_dir>/research_notes.md`." The writer told the editor: "Please read `draft_section_3.md`." **Agents cannot work on files they don't know about.** The convention of `draft_section_N.md` / `final_section_N.md` was essential — it let the editor know exactly which file to read without the writer having to explain the naming scheme each time.

### 10.10 Shutdown Is a Protocol, Not an Assumption

Agents do not terminate when they run out of tasks. They go idle and wait. The orchestrator must explicitly issue `shutdown_request` messages. Agents then confirm with `shutdown_approved` before terminating. This handshake ensures agents don't terminate mid-task and that the orchestrator knows exactly when each agent has stopped. **Never assume agents have shut down — wait for the `teammate_terminated` system message.**

---

## 11. What Could Go Wrong — Edge Cases Encountered

### Edge Case 1: Background Agent Raced the Team Researcher

**Problem:** A background agent written before `TeamCreate` was invoked completed and wrote `research_notes.md` at the same time the team's researcher was still reading source files. There was risk of the team researcher overwriting the complete file with a partial one.

**Resolution:** The orchestrator received the background agent's completion notification, immediately read `research_notes.md`, confirmed it was complete (681 lines, well-structured), and manually closed Task #1 before the team researcher could overwrite it. The team researcher received a "verify and hand off" message that told it not to duplicate effort.

**Lesson:** When background agents and team agents work on the same files, the orchestrator must actively manage race conditions.

---

### Edge Case 2: Wrong Agent Assigned to Task #4

**Problem:** The orchestrator assigned PDF assembly (Python/ReportLab work) to the researcher, which was a research-specialised agent. The researcher correctly identified it was the wrong capability match and pushed back.

**Resolution:** The orchestrator spawned a `builder` subagent specifically for the PDF task. This took seconds and produced the correct result.

**Lesson:** Agent role prompts establish capability, but task assignment still needs to match that capability. If an agent pushes back on a task for capability reasons, trust the pushback and re-route.

---

### Edge Case 3: Researcher and Builder Both Assembled the PDF

**Problem:** After the orchestrator assigned Task #4 to the researcher, it also sent a detailed assembly brief. Despite pushing back, the researcher apparently also attempted an assembly (producing an 82-page / 208 KB version). The builder produced a different version (84 pages / 211 KB). Both ran, with the builder's version being the final file on disk (ran later).

**Resolution:** The builder's version was on disk and was the correct deliverable. The duplication was benign but wasteful.

**Lesson:** Once a task is re-assigned, send a clear "do NOT proceed" message to the original assignee. Ambiguous assignment instructions can lead to duplicate effort.

---

### Edge Case 4: Researcher Required Two Shutdown Requests

**Problem:** The researcher's first `shutdown_request` arrived while it was in an `interrupted` idle state (mid-task processing). It did not cleanly acknowledge the first request.

**Resolution:** The orchestrator sent a second `shutdown_request`. The researcher then confirmed `shutdown_approved` and terminated cleanly.

**Lesson:** Some agents may miss shutdown requests if they arrive during certain idle substates. Always wait for `teammate_terminated` system confirmation — don't assume shutdown succeeded after sending the request.

---

### Edge Case 5: Writer Sent Multiple "All 8 Done" Messages

**Problem:** After Section 8 was revised, the writer sent two messages to the editor:
- "Revised Section 8 done — all 8 sections finalized"
- "All 8 sections finalized and approved"

This was slightly redundant and could have confused the editor about whether to re-read Section 8 or just acknowledge.

**Resolution:** The editor correctly interpreted both messages as completion signals and sent the "all done" message to the researcher without re-reading anything.

**Lesson:** In iterative loops, be precise about what each message signals. "Done with revision" and "all done with the project" should be distinct messages with distinct semantics.

---

## 12. Summary Statistics

### Team Composition

| Agent | Type | Tasks | Status |
|---|---|---|---|
| Orchestrator | Root session | Architecture + monitoring | Active throughout |
| Researcher | General-purpose | Task #1 | ✅ Shut down |
| Writer | General-purpose | Task #2 | ✅ Shut down |
| Editor | General-purpose | Task #3 | ✅ Shut down |
| Builder | Builder subagent | Task #4 (de facto) | ✅ Completed (no shutdown needed) |
| Background agent | General-purpose | Pre-team research | ✅ Completed |

### Task Metrics

| Metric | Value |
|---|---|
| Tasks created | 4 |
| Task dependency edges | 4 |
| Task completions | 4 |
| Orchestrator manual task updates | 3 |
| Agent self-reported completions | 2 (writer + editor) |

### Communication Metrics

| Metric | Value |
|---|---|
| Total `SendMessage` calls | ~39 |
| Peer-to-peer messages (writer ↔ editor) | ~24 |
| Orchestrator → Agent messages | ~12 |
| Agent → Orchestrator messages | ~3 |
| `idle_notification` events received | ~40+ |
| Idle events requiring orchestrator action | 3 |
| Shutdown handshakes | 3 |

### Content Metrics

| Metric | Value |
|---|---|
| Source files read | 4 |
| Web searches conducted | 6 |
| Research notes produced | 681 lines / 40 KB |
| Draft sections written | 8 |
| Final sections written | 8 |
| Total intermediate markdown | ~105 KB across 17 files |
| Editorial cycles | 16 (2 per section) |
| Sections requiring 3+ rounds | 0 |

### Final Deliverable

| Metric | Value |
|---|---|
| PDF pages | 84 |
| PDF file size | 211 KB |
| Part I (Reference Guide) | p.1–51 |
| Part II (The Article) | p.52–84 |
| Total elapsed time | ~45 minutes |
| Orchestrator interventions | 3 |

---

*Documentation produced by Claude (Sonnet 4.6) — March 2026*
*Project: Munger Latticework Mental Models — `munger-article-team`*
