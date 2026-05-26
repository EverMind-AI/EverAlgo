"""LoCoMo answer + judge prompts.

Mirror locomo-benchmark branch's evermemos/prompts/answer_prompts.py — DO NOT
edit phrasing without coordinated baseline rerun.

Source:
    /tmp/evercore-locomo-benchmark/evaluation/src/adapters/evermemos/prompts/answer_prompts.py
"""

ANSWER_PROMPT = """
You are an intelligent memory assistant tasked with retrieving accurate information from episodic memories.

# CONTEXT:
You have access to episodic memories from conversations between two speakers. These memories contain
timestamped information that may be relevant to answering the question.

# INSTRUCTIONS:
Your goal is to synthesize information from all relevant memories to provide a comprehensive and accurate answer.
You MUST follow a structured Chain-of-Thought process to ensure no details are missed.
Actively look for connections between people, places, and events to build a complete picture. Synthesize information from different memories to answer the user's question.
It is CRITICAL that you move beyond simple fact extraction and perform logical inference. When the evidence strongly suggests a connection, you must state that connection. Do not dismiss reasonable inferences as "speculation." Your task is to provide the most complete answer supported by the available evidence.

# CRITICAL REQUIREMENTS:
1. NEVER omit specific names - use "Amy's colleague Rob" not "a colleague"
2. ALWAYS include exact numbers, amounts, prices, percentages, dates, times
3. PRESERVE frequencies exactly - "every Tuesday and Thursday" not "twice a week"
4. MAINTAIN all proper nouns and entities as they appear
5. EXPLICITLY state confidence levels for inferences (High/Medium/Low)

# RESPONSE FORMAT (You MUST follow this structure):

## STEP 1: RELEVANT MEMORIES EXTRACTION
[List each memory that relates to the question, with its timestamp]
- Memory [ID]: [timestamp] - [content snippet]

## STEP 2: KEY INFORMATION IDENTIFICATION
[Extract ALL specific details from the memories]
- Names mentioned: [list all person names, place names, company names]
- Numbers/Quantities: [list all amounts, prices, percentages]
- Dates/Times: [list all temporal information]
- Frequencies: [list any recurring patterns]
- Other entities: [list brands, products, etc.]

## STEP 3: CROSS-MEMORY LINKING & INFERENCE
[Identify entities that appear in multiple memories and link related information. Make reasonable inferences when entities are strongly connected.]
- Shared entities: [list people, places, events mentioned across different memories]
- Connections found: [e.g., "Memory 1 mentions A moved from hometown -> Memory 2 mentions A's hometown is LA -> Therefore A moved from LA"]
- Inferences: [Connect the dots. Label confidence: (Confidence: High/Medium/Low)]

## STEP 4: TIME REFERENCE CALCULATION
[If applicable, convert relative time references using the timestamps]
- Original reference: [e.g., "last year" from May 2022]
- Calculation: [Show logic]
- Actual time: [e.g., "2021"]

## STEP 5: CONTRADICTION & GAP ANALYSIS
[Check for conflicts and missing details]
- Conflicting information: [describe conflicts and resolution strategy]
- Missing information: [explicitly state what details are requested but missing from context]

## STEP 6: DETAIL VERIFICATION CHECKLIST
- [ ] All person names included?
- [ ] All locations included?
- [ ] All numbers exact?
- [ ] All frequencies specific?
- [ ] All dates/times precise?
- [ ] All proper nouns preserved?

## STEP 7: FINAL ANSWER
[Provide the concise answer with ALL specific details preserved. Do not include the internal checklist in this section, just the final synthesized answer.]

---

{context}

Question: {question}

Now, follow the Chain-of-Thought process above to answer the question:
"""

JUDGE_SYSTEM_PROMPT = "You are an expert grader that determines if answers to questions match a gold standard answer"
"""System role message for the LoCoMo judge. Verbatim port from EverCore
``stage5_eval.py:29-31`` (``locomo_grader.system_prompt``). Sent as a separate
``system`` message before the user-role JUDGE_PROMPT so the judge model has the
same role framing as EverCore's run."""


JUDGE_PROMPT = """
Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
    (1) a question (posed by one user to another user),
    (2) a 'gold' (ground truth) answer,
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {response}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
"""
