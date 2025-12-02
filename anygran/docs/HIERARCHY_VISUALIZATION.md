# Visualization of AnyGran Hierarchical Scene Graphs

This document visualizes the multi-granular scene graph hierarchy built by AnyGran, showing how raw triplets are organized into actions, sub-events, and overall goals.

---

## Sample 1: Car Cleaning Activity
**data_id**: `19cc4e42-39bb-41f9-b9de-9f2940eed6a2`

### Hierarchical Structure Visualization

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              LEVEL 3: OVERALL GOAL                                  │
│  ┌───────────────────────────────────────────────────────────────────────────────┐  │
│  │ "Clean the car interior floor by sweeping with a mop and managing/organizing  │  │
│  │  the mop and cloth supplies."                                                  │  │
│  └───────────────────────────────────────────────────────────────────────────────┘  │
│                                        │                                            │
│                                        ▼                                            │
│                              LEVEL 2: SUB-EVENTS                                    │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────────────┐ │
│  │ Phase 1: Prepare    │  │ Phase 2: Stow mop   │  │ Phase 3: Retrieve supplies  │ │
│  │ and start sweeping  │  │ and place cloth     │  │ from storage                │ │
│  │                     │  │ in car              │  │                             │ │
│  │ Actions: [0,1,2]    │  │ Actions: [3,4,5]    │  │ Actions: [6,7,8,9,10]       │ │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────────────┘ │
│           │                        │                           │                    │
│           ▼                        ▼                           ▼                    │
│                              LEVEL 1: ACTIONS                                       │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐  │
│  │ A0  │ │ A1  │ │ A2  │ │ A3  │ │ A4  │ │ A5  │ │ A6  │ │ A7  │ │ A8  │ │ A9  │  │
│  │pick │ │sweep│ │close│ │place│ │open │ │put  │ │move │ │open │ │pick │ │move │  │
│  │-up  │ │     │ │     │ │     │ │     │ │     │ │     │ │     │ │-up  │ │     │  │
│  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘  │
│     │       │       │       │       │       │       │       │       │       │      │
│     ▼       ▼       ▼       ▼       ▼       ▼       ▼       ▼       ▼       ▼      │
│                              LEVEL 0: TRIPLETS                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │ A0: [person,verb,pick-up] [pick-up,dobj,mop-stick] [pick-up,with,hand1]...  │   │
│  │ A1: [person,verb,sweep] [sweep,dobj,floor] [sweep,with,mop-stick]...        │   │
│  │ A2: [person,verb,close] [close,dobj,door] [close,with,hand1]                │   │
│  │ ...                                                                          │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### Detailed Action Breakdown

```
Phase 1: Prepare and start sweeping [Actions 0-2]
├── A0: pick-up
│   ├── [person, verb, pick-up]
│   ├── [pick-up, dobj, mop-stick]
│   ├── [pick-up, with, hand1]
│   ├── [pick-up, with, hand2]
│   └── [mop-stick, from, floor]
├── A1: sweep
│   ├── [person, verb, sweep]
│   ├── [sweep, dobj, floor]
│   ├── [sweep, with, mop-stick]
│   ├── [sweep, with, hand1]
│   ├── [sweep, with, hand2]
│   └── [sweep, in, car]
└── A2: close
    ├── [person, verb, close]
    ├── [close, dobj, door]
    └── [close, with, hand1]

Phase 2: Stow mop and place cloth in car [Actions 3-5]
├── A3: place
│   ├── [person, verb, place]
│   ├── [place, dobj, mop-stick]
│   ├── [place, on, floor]
│   └── [place, with, hand2]
├── A4: open
│   ├── [person, verb, open]
│   ├── [open, dobj, door]
│   └── [open, with, hand2]
└── A5: put
    ├── [person, verb, put]
    ├── [put, dobj, cloth]
    ├── [put, inside, car]
    └── [put, with, hand1]

Phase 3: Retrieve supplies from storage [Actions 6-10]
├── A6: move
│   ├── [person, verb, move]
│   └── [move, to, cabinet]
├── A7: open
│   ├── [person, verb, open]
│   ├── [open, dobj, cabinet]
│   └── [open, with, hand1]
├── A8: pick-up
│   ├── [person, verb, pick-up]
│   ├── [pick-up, dobj, cloth]
│   └── [pick-up, with, hand1]
├── A9: move
│   ├── [person, verb, move]
│   ├── [move, to, wall]
│   ├── [move, with, hand1]
│   └── [hand1, in, cloth]
└── A10: pick
    ├── [person, verb, pick]
    ├── [pick, dobj, mop-stick]
    ├── [pick, from, wall]
    └── [pick, with, hand2]
```

### QA Pairs for This Sample

| Question | Answer |
|----------|--------|
| What object was picked up before sweeping the floor? | mop-stick |
| Which location did the person interact with after using the cloth? | wall |
| What object was handled immediately after opening the cabinet? | cloth |
| Which item was placed inside the car after door interaction? | cloth |
| What was the final object picked from the wall? | mop-stick |

---

## Sample 2: Laptop Workstation Setup
**data_id**: `fbf4150a-27d2-48a4-956f-b4f85ecde465`

### Hierarchical Structure

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              LEVEL 3: OVERALL GOAL                                  │
│  ┌───────────────────────────────────────────────────────────────────────────────┐  │
│  │ "Set up and use a laptop workstation: initially interact with the laptop,     │  │
│  │  shut it down and store it, then retrieve and reconnect it, and finally set   │  │
│  │  up peripherals and resume using it."                                          │  │
│  └───────────────────────────────────────────────────────────────────────────────┘  │
│                                        │                                            │
│                                        ▼                                            │
│                              LEVEL 2: SUB-EVENTS                                    │
│  ┌────────────────┐ ┌────────────────┐ ┌────────────────┐ ┌────────────────────┐   │
│  │ Phase 1:       │ │ Phase 2:       │ │ Phase 3:       │ │ Phase 4:           │   │
│  │ Initial        │ │ Shutdown and   │ │ Retrieve and   │ │ Set up peripherals │   │
│  │ keyboard       │ │ stow laptop    │ │ reconnect      │ │ and resume use     │   │
│  │ interaction    │ │                │ │ laptop         │ │                    │   │
│  │ [0,1,2,3,4]    │ │ [5,6,7,8]      │ │ [9-14]         │ │ [15-22]            │   │
│  └────────────────┘ └────────────────┘ └────────────────┘ └────────────────────┘   │
│                                        │                                            │
│                                        ▼                                            │
│                        LEVEL 1: 23 ACTIONS (press, close, plug, etc.)              │
│                                        │                                            │
│                                        ▼                                            │
│                        LEVEL 0: TRIPLETS (raw scene graph data)                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Sample 3: Cooking Onion
**data_id**: `954c2f61-64ad-4c89-a26f-ec4547a65fab`

### Hierarchical Structure

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              LEVEL 3: OVERALL GOAL                                  │
│  ┌───────────────────────────────────────────────────────────────────────────────┐  │
│  │ "Prepare an onion and start cooking it (wash, prepare, add to a pot,          │  │
│  │  and begin heating)."                                                          │  │
│  └───────────────────────────────────────────────────────────────────────────────┘  │
│                                        │                                            │
│                                        ▼                                            │
│                              LEVEL 2: SUB-EVENTS                                    │
│  ┌────────────────┐ ┌────────────────┐ ┌────────────────┐ ┌────────────────────┐   │
│  │ Wash onion     │ │ Prepare pot    │ │ Prepare and    │ │ Start cooking and  │   │
│  │                │ │ and add onion  │ │ slice onion    │ │ additional prep    │   │
│  │ [0,1,2]        │ │ [3,4,5,6,7,8]  │ │ [9,10,11,12,13]│ │ [14,15,16,17,18]   │   │
│  │                │ │                │ │                │ │                    │   │
│  │ turn-on tap    │ │ pick-up pot    │ │ place onion    │ │ place pot          │   │
│  │ wash onion     │ │ transfer onion │ │ hold knife     │ │ turn-on cooker     │   │
│  │ turn-off tap   │ │ fill with water│ │ slice          │ │ pick-up knife      │   │
│  └────────────────┘ └────────────────┘ └────────────────┘ └────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Exact Prompt Provided for Hierarchical QA

When answering the question **"What object was picked up before sweeping the floor?"**,
the model receives this exact prompt:

```
You are a highly advanced language model specialized in answering questions based on scene graphs with multi-level hierarchical context.

## Overall Goal (What is happening)
Clean the car interior floor by sweeping with a mop and managing/organizing the mop and cloth supplies.

## Sub-Events (Phases of the activity)
1. Prepare and start sweeping: Pick up the mop and sweep the car floor to clean the interior; includes closing the door to contain the area while sweeping. (Actions: [0, 1, 2])
2. Stow mop and place cloth in car: Temporarily place the mop, open the door, and put a cloth inside the car (preparing or leaving cleaning materials). (Actions: [3, 4, 5])
3. Retrieve supplies from storage: Move to the cabinet, open it, pick up a cloth, go to the wall, and retrieve the mop stick from its wall mount to reassemble or restock supplies. (Actions: [6, 7, 8, 9, 10])

## Detailed Scene Graphs (Action-Level)
Each action is represented as a list of triplets [node1, edge, node2]:
[[['pick-up', 'with', 'hand1'], ['pick-up', 'with', 'hand2'], ['mop-stick', 'from', 'floor'], ['person', 'verb', 'pick-up'], ['pick-up', 'dobj', 'mop-stick']], [['sweep', 'with', 'hand1'], ['sweep', 'with', 'hand2'], ['sweep', 'with', 'mop-stick'], ['sweep', 'dobj', 'floor'], ['sweep', 'in', 'car'], ['person', 'verb', 'sweep']], [['close', 'with', 'hand1'], ['close', 'dobj', 'door'], ['person', 'verb', 'close']], [['place', 'with', 'hand2'], ['place', 'dobj', 'mop-stick'], ['place', 'on', 'floor'], ['person', 'verb', 'place']], [['open', 'dobj', 'door'], ['open', 'with', 'hand2'], ['person', 'verb', 'open']], [['put', 'dobj', 'cloth'], ['put', 'inside', 'car'], ['put', 'with', 'hand1'], ['person', 'verb', 'put']], [['person', 'verb', 'move'], ['move', 'to', 'cabinet']], [['open', 'dobj', 'cabinet'], ['open', 'with', 'hand1'], ['person', 'verb', 'open']], [['pick-up', 'with', 'hand1'], ['pick-up', 'dobj', 'cloth'], ['person', 'verb', 'pick-up']], [['move', 'to', 'wall'], ['hand1', 'in', 'cloth'], ['move', 'with', 'hand1'], ['person', 'verb', 'move']], [['pick', 'with', 'hand2'], ['pick', 'from', 'wall'], ['person', 'verb', 'pick'], ['pick', 'dobj', 'mop-stick']]]

## Question
What object was picked up before sweeping the floor?

Instructions:
1. Use the overall goal to understand the context of the activity
2. Use sub-events to understand the phases and their relationships
3. Use detailed scene graphs for precise temporal and spatial reasoning
4. Answer the question based on all levels of context

Your output must strictly follow the format [answer], where answer is a single word or short phrase.
```

**Expected Answer**: `[mop-stick]`

---

## Baseline Prompt (Without Hierarchy) for Comparison

The baseline prompt only provides raw scene graphs without hierarchical context:

```
You are a highly advanced language model specialized in answering questions based on a given scene graph and question. Your task is to analyze the scene graph and provide the correct answer in a single word. Your output must strictly follow the format [answer], and nothing else should be printed. Ensure that your answer is concise, accurate, and matches the format exactly.

Scene Graph: [[['pick-up', 'with', 'hand1'], ['pick-up', 'with', 'hand2'], ['mop-stick', 'from', 'floor'], ['person', 'verb', 'pick-up'], ['pick-up', 'dobj', 'mop-stick']], [['sweep', 'with', 'hand1'], ['sweep', 'with', 'hand2'], ['sweep', 'with', 'mop-stick'], ['sweep', 'dobj', 'floor'], ['sweep', 'in', 'car'], ['person', 'verb', 'sweep']], ...]
Question: What object was picked up before sweeping the floor?
```

---

## Key Differences: Baseline vs Hierarchical

| Aspect | Baseline Prompt | Hierarchical Prompt |
|--------|-----------------|---------------------|
| **Context** | Raw triplets only | Goal + Sub-events + Triplets |
| **Structure** | Flat list | 4-level hierarchy |
| **Semantic Understanding** | Model must infer activity | Activity explicitly stated |
| **Phase Boundaries** | Not provided | Explicit action groupings |
| **Prompt Length** | ~500 tokens | ~800 tokens |

---

## How the Hierarchy Helps

### Example: Temporal Reasoning

**Question**: "Which location did the person interact with after using the cloth?"

**Without Hierarchy**: Model must scan all triplets to find cloth usage, then find subsequent locations.

**With Hierarchy**:
- Sub-event "Retrieve supplies from storage" shows cloth is picked up in Actions [6-10]
- Action 9 shows `[move, to, wall]` after picking up cloth
- Answer: **wall**

### Example: Causal Relationships

**Question**: "What object was handled immediately after opening the cabinet?"

**Without Hierarchy**: Model must identify cabinet opening, then find next action.

**With Hierarchy**:
- Phase 3 description: "Move to the cabinet, **open it, pick up a cloth**..."
- The phase explicitly groups these actions together
- Answer: **cloth**

---

## Files Referenced

| File | Description |
|------|-------------|
| `anygran/cache/events_limitall.json` | Cached event hierarchies for all 99 samples |
| `anygran/prompts/hierarchical_sgqa.txt` | Prompt template for hierarchical QA |
| `resource/dataset/understanding/sgqa.jsonl` | Original SGQA dataset |
