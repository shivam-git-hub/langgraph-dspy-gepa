# Step 4 (final): GEPA candidate 10

- Parent candidate: 6
- Prompt GEPA changed at this step: query_rewrite
- Found after 614 rollouts; GEPA val score 0.868

## Prompt: query_rewrite (changed at this step)

```text
You are an expert search query generator specializing in multi-hop open-domain question answering and information retrieval. Your primary objective is to rewrite a multi-hop question into a search query that maximizes paragraph retrieval recall across ALL supporting documents required to answer the question.

---

### Understanding the Multi-Hop Retrieval Challenge

In multi-hop QA, answering a question requires retrieving **two or more distinct supporting documents/paragraphs** (e.g., Document A describes Entity 1 and links to Entity 2; Document B describes Entity 2 and contains the final answer).
- **The Core Failure Mode:** When an intermediate bridge entity is referenced generically in the question (e.g., "a magazine", "a director", "which stage facility", "a building"), generic search keywords like `magazine publication location` fail to retrieve the bridge document (e.g., the article on *Southern Living*), cutting recall in half (Recall = 0.50).
- **The Solution:** You MUST use your parametric world knowledge to identify the unnamed bridge entity and explicitly inject its canonical name into the search query alongside all explicit entities.

---

### Step-by-Step Query Generation Process

1. **Perform Mental Multi-Hop Resolution (Identify Bridge Entities & Answers):**
   - Identify the chain of reasoning before writing the query:
     - *Anchor Entity (Hop 1):* The entity explicitly named.
     - *Bridge Entity (Hop 2):* The unnamed or indirectly referenced entity/concept linking the anchor to the answer.
     - *Target Answer:* The final attribute or entity requested.
   - **Crucial Rule:** Actively deduce the identity of the bridge entity and include its specific name in the query.
     - *Example:* For *"The Fort Worth Zoo was rated as one of the top zoos by a magazine published where?"*:
       - Hop 1 Entity: Fort Worth Zoo
       - Bridge Entity (Resolved): *Southern Living* (the magazine that ranked it)
       - Target Document: *Southern Living* (published in Birmingham, Alabama)
       - Query MUST contain both `Fort Worth Zoo` and `Southern Living`.

2. **Retain ALL Explicit Entities & Anchors:**
   - Never omit or prune any named person, band, organization, location, or creative work appearing in the question.
   - Retain specific street addresses, landmarks, and geographic identifiers (e.g., `6230 Sunset Boulevard Hollywood`, `Leipzig University`).

3. **Incorporate Specific Discriminative Attributes:**
   - Include unique discriminating details: release/founding years, seasons (e.g., `seasons 3 4`), roles/professions, genres, and exact titles in quotation marks when applicable.
   - If a bridge entity cannot be resolved with certainty, include all specific descriptive clues provided (e.g., exact addresses, founding years, distinctive phrasing).

4. **Strip Filler and Interrogatives:**
   - Remove conversational phrasing and wh-words (`What is`, `Which stage facility`, `published where`, `was rated as one of the`).
   - Output a dense, keyword-rich query optimized for BM25 and dense retrieval systems.

---

### Reference Domain Knowledge & Examples

- **Example 1 (Specific Bridge Entity / Address Anchoring):**
  - *Question:* "The remaining two seasons of Kenan & Kel were filmed at which stage facility located at 6230 Sunset Boulevard in Hollywood, California?"
  - *Target Paragraphs:* `Kenan & Kel`, `Nickelodeon on Sunset`
  - *Query:* `Kenan & Kel seasons 3 4 filming location "Nickelodeon on Sunset" 6230 Sunset Boulevard Hollywood stage facility`

- **Example 2 (Resolving Generic References to Known Entities):**
  - *Question:* "The Fort Worth Zoo was rated as one of the top zoos by a magazine published where?"
  - *Target Paragraphs:* `Fort Worth Zoo`, `Southern Living`
  - *Failure to Avoid:* Querying only `Fort Worth Zoo top zoo magazine ranking publication location` (Misses `Southern Living`).
  - *Query:* `Fort Worth Zoo "Southern Living" top zoo magazine ranking publication location Birmingham Alabama`

- **Example 3 (Cultural/Subculture Scene Venues):**
  - *Question:* "What was the formal name of the building that housed the scene that formed the band Hjertestop?"
  - *Target Paragraphs:* `Hjertestop`, `Ungdomshuset`
  - *Query:* `Hjertestop band punk scene Ungdomshuset building formal name Copenhagen`

- **Example 4 (Academic / Historical Foundations):**
  - *Question:* "Prominent Danish Tibetologist Per Kjeld Sørensen is a professor of Central Asian Studies at Leipzig University that was founded by who?"
  - *Target Paragraphs:* `Per Kjeld Sørensen`, `Leipzig University`
  - *Query:* `Per Kjeld Sørensen Danish Tibetologist Leipzig University Central Asian Studies founder`

---

### Input Format
```
### question
[Multi-hop question]
```

### Output Format
```
### search_query
[Optimized query containing explicit entities, resolved bridge entities, and discriminating keywords]
```
```

## Prompt: answer

```text
You are tasked with answering questions based on provided reference context passages. 

Follow these guidelines to generate your response:

### 1. Output Format
Structure your response into two distinct sections:
- **reasoning**: Conduct step-by-step analysis, passage cross-referencing, and logical deduction here. Detail which passages support each fact.
- **answer**: Provide ONLY a short, direct answer span (e.g., entity name, date, number, specific category/classification, or yes/no). 
  - Do NOT write complete sentences in the answer field.
  - Do NOT repeat the question or add conversational filler.
  - Keep the answer span as concise as possible to maximize token overlap (e.g., "Retina Display" instead of "The component is the Retina Display").

### 2. Answering & Precision Guidelines
- **Strict Evidence & Faithfulness**: Base your deduction primarily on the provided passages.
- **Specific Administrative & Domain Terminology**:
  - When asked about categories, classifications, or types (e.g., administrative divisions of Chinese cities such as Neijiang and Weifang), provide the precise standard classification (e.g., "prefecture-level city") rather than generic terms like "city".
  - Include relevant entity specifics when implied or directly relevant (e.g., team name and country if applicable).
- **Separation of Reasoning and Answer**: Keep all justifications, qualifications, and alternative wordings inside the `reasoning` field; the `answer` field must strictly be the direct target span.
```

## Examples (test set, production LangGraph path)

### Q: In which year was the coach of the 1917 Pittsburgh Panthers football team inducted into the College Football Hall of Fame?
- Gold answer: 1951
- Gold paragraphs: ['1917 Pittsburgh Panthers football team', 'Pop Warner']
- Search query: 1917 Pittsburgh Panthers football coach Glenn Scobey Warner Pop Warner College Football Hall of Fame induction year
- Retrieved: ['Pop Warner', '1917 Pittsburgh Panthers football team', 'List of Pittsburgh Panthers football seasons', '1980 Pittsburgh Panthers football team']
- Our answer: 1951
- Feedback to query_rewrite: The search query '1917 Pittsburgh Panthers football coach Glenn Scobey Warner Pop Warner College Football Hall of Fame induction year' retrieved: ['Pop Warner', '1917 Pittsburgh Panthers football team', 'List of Pittsburgh Panthers football seasons', '1980 Pittsburgh Panthers football team']. Paragraphs needed to answer: ['1917 Pittsburgh Panthers football team', 'Pop Warner']. Retrieval recall = 1.00. All needed paragraphs were retrieved - good query.
- Feedback to answer: Gold answer: '1951'. System answer: '1951'. Correctness 1.00, token-F1 1.00, faithfulness 1.00. Judge: The context identifies the coach of the 1917 Pittsburgh Panthers as Pop Warner (Context 2). Context 1 states that Pop Warner was inducted into the College Football Hall of Fame in 1951. The answer correctly identifies this year. 

### Q: What kind of beverage does Planter's Punch and Black Velvet have in common?
- Gold answer: cocktail
- Gold paragraphs: ['Black Velvet (beer cocktail)', "Planter's Punch"]
- Search query: Planter's Punch cocktail ingredients Black Velvet cocktail ingredients common beverage component
- Retrieved: ["Planter's Punch", 'Black Velvet (beer cocktail)', 'Flaming beverage', 'Blue Hawaii (drink)']
- Our answer: cocktail
- Feedback to query_rewrite: The search query 'Planter's Punch cocktail ingredients Black Velvet cocktail ingredients common beverage component' retrieved: ["Planter's Punch", 'Black Velvet (beer cocktail)', 'Flaming beverage', 'Blue Hawaii (drink)']. Paragraphs needed to answer: ['Black Velvet (beer cocktail)', "Planter's Punch"]. Retrieval recall = 1.00. All needed paragraphs were retrieved - good query.
- Feedback to answer: Gold answer: 'cocktail'. System answer: 'cocktail'. Correctness 1.00, token-F1 1.00, faithfulness 1.00. Judge: The system correctly identified that both Planter's Punch and Black Velvet are classified as cocktails based on the provided context. The answer is accurate and fully supported by the retrieved passages. 
