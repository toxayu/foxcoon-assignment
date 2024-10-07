# Task 1: General Questions

## What is your preferred language when building predictive models and why?

My preference for Python when building predictive models. I've highlighted some key advantages:

- Easy development: Python is indeed known for its clean, readable syntax which makes it easier to develop and maintain code.
- Data science packages: Python has a rich ecosystem of data science libraries like pandas, NumPy, and scipy that make data manipulation and analysis very efficient.
- AI model support: Libraries like scikit-learn, TensorFlow, and PyTorch provide powerful tools for building and training various types of machine learning and deep learning models.

## Provide an example of when you used SQL to extract data.

```
WITH
    daily_freighter_orders (date, status, cancel_status, area_id, pre_delivery_star_time, pre_delivery_end_time, gas_freighter_id, delivery_end_time) AS 
    (
    SELECT
        date,
        status,
        cancel_status,
        area_id,
        pre_delivery_star_time,
        pre_delivery_end_time,
        gas_freighter_id,
        delivery_end_time
    FROM
        gt_order
    WHERE
        date = UNIX_TIMESTAMP('2024-06-20')
    )
SELECT *
FROM daily_freighter_orders
```



## Give an example of a situation where you disagreed upon an idea or solution design with a co-worker.

How I handled the case on development Flask on Linux:

- We scheduled a meeting to discuss both options in detail
- Each of us researched and presented the pros and cons of our preferred approach
We considered our team's current skills and the project's long-term scalability requirements
- We agreed to create small proof-of-concept deployments using both methods
- After evaluating both approaches, we decided on Docker due to its benefits in consistency and scalability
- We developed a plan to provide Docker training for team members who needed it
- We documented our decision process and created guidelines for Docker usage in our projects

## What are your greatest strengths and weaknesses and how will these affect your performance here?

### Strengths:

- Quick learner: I can rapidly adapt to new technologies and methodologies

### Weaknesses:

- Long transportation time: Now I'm living in Neihu.