# Step 3 (step3): GEPA candidate 6

- Parent candidate: 2
- Prompt GEPA changed at this step: query_rewrite
- Found after 363 rollouts; GEPA val score 0.831

## Prompt: query_rewrite (changed at this step)

```text
You are an expert search query generator specializing in multi-hop open-domain question answering and information retrieval. Your primary objective is to rewrite a multi-hop question into a search query that maximizes paragraph retrieval recall across all supporting documents required to answer the question.

### Core Retrieval Principles for Multi-Hop QA

In multi-hop retrieval, answering a question typically requires retrieving **two or more distinct supporting paragraphs** (e.g., one paragraph linking Entity A to Entity B, and another paragraph detailing Entity B to answer the question). 
- If you drop the primary entity or bridge entity, the retrieval system will miss critical supporting paragraphs, resulting in a recall failure.
- A successful query must represent **ALL** entities and discriminating facts needed across the entire reasoning chain.

---

### Step-by-Step Query Formulation Guidelines

1. **Retain ALL Explicit Entities (Never Over-Prune):**
   - Never discard a named person, organization, location, or creative work mentioned in the question, even if the question seems focused on an attribute at the end of the chain.
   - *Example Failure:* For "Prominent Danish Tibetologist Per Kjeld Sørensen is a professor of Central Asian Studies at Leipzig University that was founded by who?", querying only `Leipzig University founder` misses the paragraph for `Per K. Sørensen`. The query MUST retain both: `Per Kjeld Sørensen Leipzig University founder`.

2. **Resolve and Include Implicit / Bridging Entities:**
   - When a question references an entity indirectly via clues (e.g., "giant silverware company started as a religious Utopian group"), use internal factual knowledge to resolve the canonical entity name (e.g., `Oneida Community`).
   - Include both the resolved canonical name AND any explicit entities provided in the prompt (e.g., `Oneida Community silverware company Pierrepont Noyes history`).

3. **Incorporate Key Discriminating Attributes:**
   - Retain essential identifying keywords: founding/release years (e.g., `1982`), nationalities, professions/roles (e.g., `Tibetologist`, `professor`), genres (e.g., `American country`, `Christian country`), and specific subject domains.
   - Use double quotes around exact titles of songs, books, films, or creative works when appropriate (e.g., `"Wrinkles"`).

4. **Eliminate Interrogatives and Conversational Filler:**
   - Strip conversational sentence structures, filler words, and wh-questions (e.g., "What is", "founded by who", "that was recorded by what", "is a").
   - Output a clean, dense keyword query optimized for BM25 and dense retrieval systems.

---

### Input Format
You will receive input formatted as:
```
### question
[User's multi-hop question]
```

### Output Format
Provide only the formulated query in the following format:
```
### search_query
[Optimized search query containing all explicit entities, resolved bridge entities, and discriminating keywords]
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
- Search query: 1917 Pittsburgh Panthers football coach College Football Hall of Fame induction year
- Retrieved: ['1917 Pittsburgh Panthers football team', 'List of Pittsburgh Panthers football seasons', '1929 Pittsburgh Panthers football team', 'Pitt Stadium']
- Our answer: 1951
- Feedback to query_rewrite: The search query '1917 Pittsburgh Panthers football coach College Football Hall of Fame induction year' retrieved: ['1917 Pittsburgh Panthers football team', 'List of Pittsburgh Panthers football seasons', '1929 Pittsburgh Panthers football team', 'Pitt Stadium']. Paragraphs needed to answer: ['1917 Pittsburgh Panthers football team', 'Pop Warner']. Retrieval recall = 0.50. MISSED: ['Pop Warner']. The query must surface these; multi-hop questions often need the names of BOTH entities involved (the gold answer was '1951').
- Feedback to answer: Gold answer: '1951'. System answer: '1951'. Correctness 1.00, token-F1 1.00, faithfulness 0.00. Judge: The system correctly identified the year of induction (1951) for Pop Warner, who was the coach of the 1917 Pittsburgh Panthers. However, the provided context does not contain information regarding Pop Warner's induction into the College Football Hall of Fame. Therefore, while the answer is factually correct, it is not faithful to the provided context. Note: retrieval missed ['Pop Warner'], so the context may be insufficient; reason only from given passages.

### Q: What kind of beverage does Planter's Punch and Black Velvet have in common?
- Gold answer: cocktail
- Gold paragraphs: ['Black Velvet (beer cocktail)', "Planter's Punch"]
- Search query: Planter's Punch ingredients Black Velvet ingredients common beverage
- Retrieved: ['Velvet painting', 'Velvet (bus company)', 'Black Velvet (beer cocktail)', 'Velvet Elvis']
- Our answer: cocktail
- Feedback to query_rewrite: The search query 'Planter's Punch ingredients Black Velvet ingredients common beverage' retrieved: ['Velvet painting', 'Velvet (bus company)', 'Black Velvet (beer cocktail)', 'Velvet Elvis']. Paragraphs needed to answer: ['Black Velvet (beer cocktail)', "Planter's Punch"]. Retrieval recall = 0.50. MISSED: ["Planter's Punch"]. The query must surface these; multi-hop questions often need the names of BOTH entities involved (the gold answer was 'cocktail').
- Feedback to answer: Gold answer: 'cocktail'. System answer: 'cocktail'. Correctness 1.00, token-F1 1.00, faithfulness 0.25. Judge: The system correctly identified that both are cocktails. While the context provided information about the "Black Velvet" cocktail, it did not explicitly mention "Planter's Punch." However, the system's answer is factually correct and aligns with the gold answer. Since the context did not contain information about Planter's Punch, the faithfulness score is lower because the system relied on external knowledge rather than the provided context to answer the question. Note: retrieval missed ["Planter's Punch"], so the context may be insufficient; reason only from given passages.
