# Recommendations for Model Enhancement

Based on error analysis of the SGQA hard benchmark, here are targeted recommendations to improve model performance.

## 1. Temporal Ordering (55 errors)

### Problem
Models struggle with questions requiring precise sequence tracking:
- "What was picked up BEFORE X?"
- "What was the LAST tool used after Y?"
- "What object was handled IMMEDIATELY after Z?"

### Recommendations

1. **Explicit Action Indexing**
   - Add action indices to scene graph representation: "Action 0: pick-up knife, Action 1: cut onion..."
   - Makes temporal relationships explicit

2. **Chain-of-Thought Prompting**
   - Add reasoning step: "First, identify all actions. Then, find the reference action X. Finally, look at adjacent actions."
   - Force step-by-step temporal analysis

3. **Reverse Search for "Before"**
   - When question asks "before X", search backwards from X
   - Current models often search forward and miss the answer

4. **Boundary Detection**
   - "Last X before Y" requires finding boundary at Y, then searching backwards
   - Explicit boundary markers in prompts

## 2. Multi-Step Reasoning (17 errors)

### Problem
Questions requiring tracking completion states:
- "Which tool was used after ALL dough kneading was completed?"
- "What was handled after the ENTIRE sequence finished?"

### Recommendations

1. **Phase-Based Grouping (EpiMine approach)**
   - Group actions into semantic phases
   - Track phase completion explicitly
   - EpiMine already shows improvement here

2. **Completion Detection**
   - Add explicit markers for repeated action completion
   - "Kneading phase: Actions 3-7 (completed)"

3. **Hierarchical Context**
   - Overall goal → Sub-events → Actions → Triplets
   - Helps identify when a sub-task is complete

## 3. Both-Hands Manipulation (9 errors)

### Problem
Questions about simultaneous hand states:
- "Which object required both hands?"
- "What was held in hand2 while hand1 was picking up X?"

### Recommendations

1. **Explicit Hand State Tracking**
   - Track hand1_holding and hand2_holding at each action
   - "Action 5: hand1=knife, hand2=onion"

2. **Both-Hands Flag**
   - Mark actions that require both hands explicitly
   - "Action 7: pick-up board (both_hands=true)"

3. **State Timeline**
   - Maintain parallel state tracks for each hand
   - Query: "At action N, what was in hand2?"

## 4. Concurrent State (2 errors)

### Problem
Questions about simultaneous events:
- "What was held WHILE the drill was operating?"
- "Which object was handled DURING brushing?"

### Recommendations

1. **Temporal Overlap Detection**
   - Identify actions that overlap in time
   - Mark concurrent actions explicitly

2. **State Snapshot**
   - At each action, capture full state: {current_action, hand1, hand2, location}
   - Query snapshots for "while/during" questions

## 5. General Improvements

### Object Disambiguation
- Similar objects (screw vs screwdriver, pot vs pan) cause confusion
- Add object descriptions or unique identifiers

### Answer Format Consistency
- Ensure single-word answers match ground truth format
- "metal-board" vs "metal board" should be handled

### Context Length Handling
- Long action sequences (>20 actions) cause degradation
- Consider chunking or summarization for very long sequences

## Implementation Priority

| Priority | Category | Potential Impact |
|----------|----------|------------------|
| 1 | Temporal Ordering | High (55 errors) |
| 2 | Multi-Step Reasoning | Medium (17 errors) |
| 3 | Both-Hands | Medium (9 errors) |
| 4 | Concurrent State | Low (2 errors) |
