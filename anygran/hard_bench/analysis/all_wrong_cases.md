# Hardest Cases: All Models Failed

These 32 questions were answered incorrectly by all 3 models (EpiMine v0, Baseline-mini, Baseline-GPT5).

## Cases

### Case 1: 860980fb

**Question:** Which object required both hands for its first manipulation after being picked up?

**Ground Truth:** `metal-board`

**Predictions:**
- EpiMine v0: `paper`
- Baseline-mini: `paper`
- Baseline-GPT5: `paper`

**Category:** both_hands

**Analysis:** Requires tracking hand1/hand2 object manipulation. All models predicted 'paper' instead of 'metal-board'

---

### Case 2: 41b8254c

**Question:** Which object was handled by both hands during the drink preparation sequence?

**Ground Truth:** `bottle`

**Predictions:**
- EpiMine v0: `spoon`
- Baseline-mini: `spoon`
- Baseline-GPT5: `spoon`

**Category:** concurrent_state

**Analysis:** Requires tracking simultaneous states. All models predicted 'spoon' instead of 'bottle'

---

### Case 3: 312da1d0

**Question:** What was the last tool used before the final water spray sequence began?

**Ground Truth:** `bottle`

**Predictions:**
- EpiMine v0: `hose`
- Baseline-mini: `hose`
- Baseline-GPT5: `hose`

**Category:** multi_step

**Analysis:** Requires multi-step reasoning and completion detection. All models predicted 'hose' instead of 'bottle'

---

### Case 4: 3db79f36

**Question:** Which object was manipulated immediately after the first sewing sequence completed?

**Ground Truth:** `scissors`

**Predictions:**
- EpiMine v0: `thread`
- Baseline-mini: `thread`
- Baseline-GPT5: `thread`

**Category:** multi_step

**Analysis:** Requires multi-step reasoning and completion detection. All models predicted 'thread' instead of 'scissors'

---

### Case 5: 3ea57fa2

**Question:** Which tool was used after all dough kneading and adjusting sequences were completed?

**Ground Truth:** `knife`

**Predictions:**
- EpiMine v0: `pan`
- Baseline-mini: `pan`
- Baseline-GPT5: `pan`

**Category:** multi_step

**Analysis:** Requires multi-step reasoning and completion detection. All models predicted 'pan' instead of 'knife'

---

### Case 6: 82dc7a9e

**Question:** What tool was used immediately before the bolt tightening sequence began?

**Ground Truth:** `paintbrush`

**Predictions:**
- EpiMine v0: `screwdriver`
- Baseline-mini: `screwdriver`
- Baseline-GPT5: `screwdriver`

**Category:** multi_step

**Analysis:** Requires multi-step reasoning and completion detection. All models predicted 'screwdriver' instead of 'paintbrush'

---

### Case 7: a0d6bd8e

**Question:** What tool was used immediately after the first water-filling action completed?

**Ground Truth:** `hose`

**Predictions:**
- EpiMine v0: `bucket`
- Baseline-mini: `bucket`
- Baseline-GPT5: `bucket`

**Category:** multi_step

**Analysis:** Requires multi-step reasoning and completion detection. All models predicted 'bucket' instead of 'hose'

---

### Case 8: f643d88a

**Question:** What object was manipulated between the two vine-cutting sequences?

**Ground Truth:** `plant`

**Predictions:**
- EpiMine v0: `shears`
- Baseline-mini: `rake`
- Baseline-GPT5: `hose`

**Category:** multi_step

**Analysis:** Requires multi-step reasoning and completion detection

---

### Case 9: 1695475c

**Question:** Which object was manipulated immediately after the second wire-cutting operation?

**Ground Truth:** `cable`

**Predictions:**
- EpiMine v0: `wire-cutter`
- Baseline-mini: `wire-cutter`
- Baseline-GPT5: `wire-cutter`

**Category:** temporal_ordering

**Analysis:** Requires precise temporal sequence tracking. All models predicted 'wire-cutter' instead of 'cable'

---

### Case 10: 312da1d0

**Question:** Which action was performed on the generator immediately after fuel was added?

**Ground Truth:** `check`

**Predictions:**
- EpiMine v0: `pick-up`
- Baseline-mini: `pick-up`
- Baseline-GPT5: `pick-up`

**Category:** temporal_ordering

**Analysis:** Requires precise temporal sequence tracking. All models predicted 'pick-up' instead of 'check'

---

### Case 11: 312da1d0

**Question:** Which object was manipulated after the tank was closed but before watering began?

**Ground Truth:** `wire`

**Predictions:**
- EpiMine v0: `hose`
- Baseline-mini: `hose`
- Baseline-GPT5: `lid`

**Category:** temporal_ordering

**Analysis:** Requires precise temporal sequence tracking

---

### Case 12: 41b8254c

**Question:** What container was accessed last before adding ice-cubes to the cup?

**Ground Truth:** `freezer`

**Predictions:**
- EpiMine v0: `tray`
- Baseline-mini: `tray`
- Baseline-GPT5: `tray`

**Category:** temporal_ordering

**Analysis:** Requires precise temporal sequence tracking. All models predicted 'tray' instead of 'freezer'

---

### Case 13: 44ae604b

**Question:** What object was manipulated with hand2 before the sponge was dipped?

**Ground Truth:** `wiper`

**Predictions:**
- EpiMine v0: `bucket`
- Baseline-mini: `bucket`
- Baseline-GPT5: `bucket`

**Category:** temporal_ordering

**Analysis:** Requires precise temporal sequence tracking. All models predicted 'bucket' instead of 'wiper'

---

### Case 14: 49374bc5

**Question:** Which tool was used immediately after putting down the rag for the first time?

**Ground Truth:** `hose`

**Predictions:**
- EpiMine v0: `rag`
- Baseline-mini: `rag`
- Baseline-GPT5: `rag`

**Category:** temporal_ordering

**Analysis:** Requires precise temporal sequence tracking. All models predicted 'rag' instead of 'hose'

---

### Case 15: 6f082d5d

**Question:** What tool was used immediately before the dough was cut into strips?

**Ground Truth:** `cutter`

**Predictions:**
- EpiMine v0: `scraper`
- Baseline-mini: `scraper`
- Baseline-GPT5: `scraper`

**Category:** temporal_ordering

**Analysis:** Requires precise temporal sequence tracking. All models predicted 'scraper' instead of 'cutter'

---

### Case 16: 6f082d5d

**Question:** Which tool was used last before the dough-strips were placed in the pan?

**Ground Truth:** `rolling-pin`

**Predictions:**
- EpiMine v0: `stove`
- Baseline-mini: `stove`
- Baseline-GPT5: `stove`

**Category:** temporal_ordering

**Analysis:** Requires precise temporal sequence tracking. All models predicted 'stove' instead of 'rolling-pin'

---

### Case 17: 950f70a4

**Question:** What was the final item cleaned before the sink was washed?

**Ground Truth:** `pan`

**Predictions:**
- EpiMine v0: `counter`
- Baseline-mini: `counter`
- Baseline-GPT5: `counter`

**Category:** temporal_ordering

**Analysis:** Requires precise temporal sequence tracking. All models predicted 'counter' instead of 'pan'

---

### Case 18: d19faa97

**Question:** What was the last object modified before the final reading state?

**Ground Truth:** `stand`

**Predictions:**
- EpiMine v0: `book`
- Baseline-mini: `book`
- Baseline-GPT5: `book`

**Category:** temporal_ordering

**Analysis:** Requires precise temporal sequence tracking. All models predicted 'book' instead of 'stand'

---

### Case 19: d4fc1fd7

**Question:** What tool was used immediately before cleaning the hand?

**Ground Truth:** `wrench`

**Predictions:**
- EpiMine v0: `tissue`
- Baseline-mini: `tissue`
- Baseline-GPT5: `tissue`

**Category:** temporal_ordering

**Analysis:** Requires precise temporal sequence tracking. All models predicted 'tissue' instead of 'wrench'

---

### Case 20: d6de6eee

**Question:** What object was handled immediately before the first nut peeling action?

**Ground Truth:** `knife`

**Predictions:**
- EpiMine v0: `nut`
- Baseline-mini: `nut`
- Baseline-GPT5: `nut`

**Category:** temporal_ordering

**Analysis:** Requires precise temporal sequence tracking. All models predicted 'nut' instead of 'knife'

---

### Case 21: d6de6eee

**Question:** What object was placed onto the cracker immediately before the final press action?

**Ground Truth:** `nut`

**Predictions:**
- EpiMine v0: `cracker`
- Baseline-mini: `cracker`
- Baseline-GPT5: `cracker`

**Category:** temporal_ordering

**Analysis:** Requires precise temporal sequence tracking. All models predicted 'cracker' instead of 'nut'

---

### Case 22: e08c9103

**Question:** What tool was picked up immediately before the grinding wheel was installed?

**Ground Truth:** `chisel`

**Predictions:**
- EpiMine v0: `wheel`
- Baseline-mini: `wheel`
- Baseline-GPT5: `wrench`

**Category:** temporal_ordering

**Analysis:** Requires precise temporal sequence tracking

---

### Case 23: e08c9103

**Question:** Which object was manipulated twice with hand1 before the final grinding operation?

**Ground Truth:** `key`

**Predictions:**
- EpiMine v0: `handle`
- Baseline-mini: `handle`
- Baseline-GPT5: `handle`

**Category:** temporal_ordering

**Analysis:** Requires precise temporal sequence tracking. All models predicted 'handle' instead of 'key'

---

### Case 24: e43f5b38

**Question:** Which tool was used last before the dough entered the oven?

**Ground Truth:** `container`

**Predictions:**
- EpiMine v0: `roller`
- Baseline-mini: `spoon`
- Baseline-GPT5: `roller`

**Category:** temporal_ordering

**Analysis:** Requires precise temporal sequence tracking

---

### Case 25: e8cf9894

**Question:** Which object was manipulated immediately before applying egg to dough-pieces?

**Ground Truth:** `brush`

**Predictions:**
- EpiMine v0: `tray`
- Baseline-mini: `tray`
- Baseline-GPT5: `tray`

**Category:** temporal_ordering

**Analysis:** Requires precise temporal sequence tracking. All models predicted 'tray' instead of 'brush'

---

### Case 26: e9be1118

**Question:** What was the last tool picked up before the final positioning action?

**Ground Truth:** `screw`

**Predictions:**
- EpiMine v0: `screwdriver`
- Baseline-mini: `screwdriver`
- Baseline-GPT5: `screwdriver`

**Category:** temporal_ordering

**Analysis:** Requires precise temporal sequence tracking. All models predicted 'screwdriver' instead of 'screw'

---

### Case 27: ea87324e

**Question:** What tool was used immediately before the wood was wiped?

**Ground Truth:** `sandpaper`

**Predictions:**
- EpiMine v0: `cloth`
- Baseline-mini: `cloth`
- Baseline-GPT5: `cloth`

**Category:** temporal_ordering

**Analysis:** Requires precise temporal sequence tracking. All models predicted 'cloth' instead of 'sandpaper'

---

### Case 28: efa59c4e

**Question:** Which tool was used in the first complete assembly cycle before the brake installation?

**Ground Truth:** `screwdriver`

**Predictions:**
- EpiMine v0: `wrench`
- Baseline-mini: `wrench`
- Baseline-GPT5: `wrench`

**Category:** temporal_ordering

**Analysis:** Requires precise temporal sequence tracking. All models predicted 'wrench' instead of 'screwdriver'

---

### Case 29: f643d88a

**Question:** Which gardening tool was used last after completing plant maintenance?

**Ground Truth:** `hose`

**Predictions:**
- EpiMine v0: `shears`
- Baseline-mini: `shears`
- Baseline-GPT5: `shears`

**Category:** temporal_ordering

**Analysis:** Requires precise temporal sequence tracking. All models predicted 'shears' instead of 'hose'

---

### Case 30: f646012d

**Question:** What object was manipulated immediately after the final screwdriver interaction?

**Ground Truth:** `cover`

**Predictions:**
- EpiMine v0: `screwdriver`
- Baseline-mini: `screwdriver`
- Baseline-GPT5: `screwdriver`

**Category:** temporal_ordering

**Analysis:** Requires precise temporal sequence tracking. All models predicted 'screwdriver' instead of 'cover'

---

### Case 31: fbf4150a

**Question:** What was the first object the person interacted with after placing the laptop?

**Ground Truth:** `charger`

**Predictions:**
- EpiMine v0: `laptop`
- Baseline-mini: `laptop`
- Baseline-GPT5: `laptop`

**Category:** temporal_ordering

**Analysis:** Requires precise temporal sequence tracking. All models predicted 'laptop' instead of 'charger'

---

### Case 32: fdb68b7b

**Question:** What object was picked up between releasing the bicycle and attaching the wheel?

**Ground Truth:** `bolt`

**Predictions:**
- EpiMine v0: `wheel`
- Baseline-mini: `nothing`
- Baseline-GPT5: `none`

**Category:** temporal_ordering

**Analysis:** Requires precise temporal sequence tracking

---

