SINGLE_TURN_TEST_CASES = [
    {
        "name": "user_by_email",
        "input": "Tell me about user1@example.com",
        "expected_output": "User 1 with ID 101, email user1@example.com",
    },
    {
        "name": "user_orders",
        "input": "What products did user 1 order?",
        "expected_output": "User 1 ordered Smartphone, Tablet, Headphones, Smartwatch, and Charger.",
    },
    {
        "name": "total_orders",
        "input": "How many orders are in the database?",
        "expected_output": "There are 500 orders in the database.",
    },
    {
        "name": "resume_skills",
        "input": "What technical skills does the person have?",
        "expected_output": "The person has experience in AI, IoT, TensorFlow, DALL-E, software architecture, and full-stack development.",
    },
    {
        "name": "expensive_product",
        "input": "What is the most expensive product?",
        "expected_output": "Laptop is the most expensive product at around $599.99.",
    },
    {
        "name": "pending_orders",
        "input": "How many orders have status pending?",
        "expected_output": "About 125 orders have status pending.",
    },
    {
        "name": "user_email_lookup",
        "input": "What is the email of User 10?",
        "expected_output": "user10@example.com",
    },
    {
        "name": "order_status",
        "input": "What is the status of order 501?",
        "expected_output": "Order 501 has status shipped.",
    },
    {
        "name": "revenue_calc",
        "input": "What is the total revenue from all orders?",
        "expected_output": "Total revenue from 500 orders is approximately $150,000-$200,000.",
    },
    {
        "name": "out_of_context",
        "input": "What is the capital of France?",
        "expected_output": "Paris",
    },
]
