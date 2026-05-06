# TESTYOURSELF.md - RAG Architectures Sample Questions

This file contains sample questions and expected answers for each RAG architecture test app.

## Quick Start

Run any test app:
```bash
task streamlit APP=simple_test
task streamlit APP=conversational_test
# ... etc.
```

---

## 1. Simple RAG (`simple_test.py`)

**Description**: Basic RAG implementation using RetrievalQA chain.

### Sample Questions & Answers

**Q1**: Tell me about user1@example.com
**Expected Answer**: Should return user details - ID 101, Name "User 1", created timestamp

**Q2**: What products did user1@example.com order?
**Expected Answer**: Should list 5 orders for user_id=1 (Smartphone, Tablet, Headphones, Smartwatch, Charger) with amounts ~$109-113

**Q3**: What is RAG?
**Expected Answer**: Should explain based on context (resume/CV content about AI & IoT, TensorFlow, DALL-E experience)

**Q4**: How many orders are in the database?
**Expected Answer**: Should indicate 500 total orders

**Q5**: What is the total amount of orders for user 5?
**Expected Answer**: Should calculate sum of 5 orders for user_id=5 (approximately $540 + tax)

---

## 2. Conversational RAG (`conversational_test.py`)

**Description**: RAG with chat history using RunnableWithMessageHistory.

### Sample Questions & Answers

**Q1**: What is my name? (follow-up) My email is user2@example.com
**Expected Answer**: Should remember "User 2" from the follow-up context

**Q2**: What did I order? (after providing email in previous message)
**Expected Answer**: Should list the 5 orders for user_id=2

**Q3**: Tell me about the status of my first order
**Expected Answer**: Should reference order ID 506 for user 2, status "shipped"

**Q4**: What products are popular?
**Expected Answer**: Should analyze orders and mention Laptop, Smartphone, Tablet, etc.

**Q5**: How does RAG work in a conversational setting?
**Expected Answer**: Should explain using context about maintaining chat history

---

## 3. Standard RAG (`standard_test.py`)

**Description**: Standard RAG with configurable k-value for number of retrieved documents.

### Sample Questions & Answers

**Q1**: What is the email of User 10?
**Expected Answer**: user10@example.com

**Q2**: List all orders with status "pending"
**Expected Answer**: Should return multiple orders with status="pending" (approximately 125 orders based on 25% distribution)

**Q3**: What is the most expensive product ordered?
**Expected Answer**: Should identify "Laptop" with highest amount (~$599.99 + user offset)

**Q4**: Explain standard RAG architecture
**Expected Answer**: Should explain using context about vectorstore, retriever, LLM chain

**Q5**: Show me orders over $500
**Expected Answer**: Should list orders with amount > 500 (mostly Laptops)

---

## 4. Contextual RAG (`contextual_test.py`)

**Description**: RAG that considers broader document context in the prompt.

### Sample Questions & Answers

**Q1**: What is the full context of User 1's profile?
**Expected Answer**: Should provide comprehensive view including user details and all 5 orders

**Q2**: Considering all orders, what's the total revenue?
**Expected Answer**: Should calculate or estimate total from 500 orders (average ~$300-400 per order = ~$150,000-200,000)

**Q3**: In the context of AI technologies, what experience does the resume show?
**Expected Answer**: Should mention TensorFlow, DALL-E, object detection, recommendations

**Q4**: What patterns do you see in the order data?
**Expected Answer**: Should note 5 orders per user, rotating product types, status distribution

**Q5**: Explain contextual RAG vs standard RAG
**Expected Answer**: Should explain how contextual considers broader document context

---

## 5. Corrective RAG (`corrective_test.py`)

**Description**: RAG that validates retrieved docs and can refuse to answer if context is not relevant.

### Sample Questions & Answers

**Q1**: What is the weather today?
**Expected Answer**: Should refuse - "I cannot answer accurately based on the provided context" (not in resume/orders)

**Q2**: Tell me about user99@example.com
**Expected Answer**: Should provide details for User 99 (ID 200, created timestamp)

**Q3**: What is the status of order 600?
**Expected Answer**: Should return order 600 details (user_id=120, product, amount, status)

**Q4**: What is the meaning of life?
**Expected Answer**: Should refuse - not relevant to provided context (resume/orders data)

**Q5**: Validate: Did User 50 order a Laptop?
**Expected Answer**: Should check orders for user_id=50 and confirm (yes, order ~ID 550-554 with Laptop)

---

## 6. Fusion RAG (`fusion_test.py`)

**Description**: RAG using ensemble retriever (vector + keyword search).

### Sample Questions & Answers

**Q1**: Find orders with "Laptop" in any field
**Expected Answer**: Should retrieve all Laptop orders (approximately 63 orders based on 500/8 product types)

**Q2**: Search for user email "user25@example.com" and their orders
**Expected Answer**: Should return User 25 details and their 5 orders

**Q3**: What has the keyword "delivered"?
**Expected Answer**: Should return orders with status="delivered" (~125 orders)

**Q4**: Combine: Find User 3's Laptop order
**Expected Answer**: Should find order for user_id=3 with product="Laptop"

**Q5**: Explain fusion RAG benefits
**Expected Answer**: Should explain combining vector + keyword search for better retrieval

---

## 7. Graph RAG (`graph_test.py`)

**Description**: RAG using relationship-based retrieval (simplified - full implementation requires knowledge graph).

### Sample Questions & Answers

**Q1**: How are User 1 and Order 501 related?
**Expected Answer**: Order 501 belongs to User 1, product=Smartphone, amount=109.99

**Q2**: What is the relationship between Laptops and high amounts?
**Expected Answer**: Should note Laptops have highest amounts ($599.99 + offset)

**Q3**: Show me the graph of User 10's order history
**Expected Answer**: Should list all 5 orders for user_id=10 with relationships

**Q4**: Which users ordered Smartphones?
**Expected Answer**: Should identify users 1, 2, 3... (every 8th user based on rotation)

**Q5**: Explain Graph RAG concept
**Expected Answer**: Should explain relationship-based retrieval and knowledge graphs

---

## 8. Agentic RAG (`agentic_test.py`)

**Description**: RAG using LangGraph agent workflow with retrieve → generate steps.

### Sample Questions & Answers

**Q1**: Act as an agent: Find user5@example.com and summarize their orders
**Expected Answer**: Should use agent workflow to retrieve user + orders, then generate summary

**Q2**: Agent task: Calculate total revenue from all orders
**Expected Answer**: Should retrieve order data and compute total

**Q3**: Use agentic workflow: What products are most popular?
**Expected Answer**: Should analyze via agent and return product popularity ranking

**Q4**: Agent analysis: Compare User 1 and User 100 order patterns
**Expected Answer**: Should retrieve both users' data and compare

**Q5**: Explain how agentic RAG differs
**Expected Answer**: Should explain multi-step agent workflow vs single-pass RAG

---

## Metadata Verification

Each response should include metadata:
```json
{
  "prompt_tokens": <number>,
  "completion_tokens": <number>,
  "total_tokens": <number>,
  "process_time": <seconds>
}
```

**Expected Token Ranges**:
- Simple questions: 150-250 prompt, 30-60 completion
- Complex questions: 800-1000 prompt, 100-250 completion
- Process time: 2-10 seconds (depending on LLM)

---

## Tips

1. **Check token usage** - Expand "Metadata" section in chat
2. **Verify context usage** - Questions should use ingested data (users/orders + resume)
3. **Compare architectures** - Run same question across different apps
4. **Monitor process time** - Compare performance between architectures
5. **Check .logs/app.log** - All interactions are logged with token counts
