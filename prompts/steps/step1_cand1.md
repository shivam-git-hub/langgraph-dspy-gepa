# Step 1 (step1): GEPA candidate 1

- Parent candidate: 0
- Prompt GEPA changed at this step: answer
- Found after 62 rollouts; GEPA val score 0.832

## Prompt: query_rewrite

```text
Rewrite the user's question into a search query.
```

## Prompt: answer (changed at this step)

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
- Search query: common ingredient in Planter's Punch and Black Velvet
- Retrieved: ['Velvet painting', 'Velvet (bus company)', 'Velvet Elvis', 'Black rose (symbolism)']
- Our answer: Not mentioned
- Feedback to query_rewrite: The search query 'common ingredient in Planter's Punch and Black Velvet' retrieved: ['Velvet painting', 'Velvet (bus company)', 'Velvet Elvis', 'Black rose (symbolism)']. Paragraphs needed to answer: ['Black Velvet (beer cocktail)', "Planter's Punch"]. Retrieval recall = 0.00. MISSED: ['Black Velvet (beer cocktail)', "Planter's Punch"]. The query must surface these; multi-hop questions often need the names of BOTH entities involved (the gold answer was 'cocktail').
- Feedback to answer: Gold answer: 'cocktail'. System answer: 'Not mentioned'. Correctness 1.00, token-F1 0.00, faithfulness 1.00. Judge: The provided context contains information about velvet paintings, a bus company, and black roses, but it does not contain any information regarding "Planter's Punch" or "Black Velvet" as beverages. Therefore, the system correctly identified that the information was not mentioned in the provided context. The answer is right but wordy: keep the reasoning in the reasoning field and make the final answer field a short, direct span (entity, date, number, yes/no). Note: retrieval missed ['Black Velvet (beer cocktail)', "Planter's Punch"], so the context may be insufficient; reason only from given passages.
