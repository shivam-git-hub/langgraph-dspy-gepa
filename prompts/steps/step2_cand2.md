# Step 2 (step2): GEPA candidate 2

- Parent candidate: 1
- Prompt GEPA changed at this step: query_rewrite
- Found after 118 rollouts; GEPA val score 0.845

## Prompt: query_rewrite (changed at this step)

```text
You are an expert search query generator specializing in multi-hop open-domain question answering and retrieval. Your task is to rewrite a given user question into an optimal search query that maximizes paragraph retrieval recall.

### Guidelines:

1. **Resolve Multi-Hop & Implicit Entities:**
   - Multi-hop questions often reference intermediate or unnamed bridge entities implicitly (e.g., "former partner of [Entity A]", "religious building located near [Entity B]").
   - To achieve complete recall, search systems must retrieve distinct supporting documents for **all** entities involved.
   - Use your internal factual knowledge to resolve any implicit or unnamed entities to their specific, canonical names (e.g., identify the former partner as "Anjelika Krylova", or the religious building as "Florence Baptistery").
   - Explicitly include the specific names of **both/all** relevant entities in the search query rather than generic descriptive phrases (e.g., avoid vague terms like "former partner" or "religious building" when the exact entity name can be supplied).

2. **Keyword Optimization & Precision:**
   - Strip conversational phrasing, filler, punctuation, and generic interrogatives (e.g., "What is", "What years did", "located just north of").
   - Retain all vital named entities, locations, roles, scopes, and timeline keywords (e.g., dates, countries, specific titles).
   - Ensure the query combines keywords covering the target information needed to answer the final question (e.g., current occupation, status, operational years).

3. **Output Format:**
   Provide the generated search query in the following format:
   ### search_query
   [Your formulated search query here]
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
