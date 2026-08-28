-- SQL Exercises: Agent Telemetry
-- Work through these in order. Each prompt is written for PostgreSQL.

-- 1. Return every column from the sessions table.
    SELECT * FROM sessions
-- 2. List the session_id, user_id, plan, and network for sessions using wifi.
    SELECT (session_id, user_id, plan, network) from sessions where network = "wifi"

-- 3. Find all LLM calls with a latency_ms greater than 5,000, showing the
--    call_id, model, latency_ms, and status, ordered from slowest to fastest.
    SELECT * FROM llm_calls where latency_ms > 5000 ORDER BY latency_ms ASC

-- 4. Return successful calls whose completion_tokens are between 100 and 500
--    inclusive, ignoring differences in the capitalization of status.
    SELECT * FROM llm_calls where latency_ms > 100 and latency_ms < 500
-- 5. Count the number of LLM calls for each status and order the result from
--    most common status to least common status.
    SELECT status, count(*) as call_count from llm_calls group by status

-- 6. For each model, calculate the total prompt_tokens, total completion_tokens,
--    and average latency_ms.

-- 7. Show each tool_name used by at least five calls and its call count.

-- 8. Find models whose average logged_cost_usd is greater than 0.001.
    

-- 9. Use an INNER JOIN to list each call_id with its session's user_id, plan,
--    and the call's model and status.

-- 10. Use a LEFT JOIN from sessions to llm_calls to count calls per session,
--     including sessions that have no calls.

-- 11. Use a RIGHT JOIN from llm_calls to model_pricing to list every priced
--     model, including models with no matching call.

-- 12. Use a FULL OUTER JOIN between sessions and llm_calls on session_id to
--     identify rows that have no match on either side.

-- 13. Use an INNER JOIN across llm_calls, sessions, and evaluation to return
--     calls evaluated with the rubric 'safety', including the user_id and score.

-- 14. Find sessions whose number of calls is greater than the average number
--     of calls per session, using a subquery.

-- 15. Return calls whose logged_cost_usd is greater than the overall average
--     logged_cost_usd, using a scalar subquery.

-- 16. Find the most expensive call for each model, including ties, using a
--     correlated subquery or a subquery joined back to llm_calls.

-- 17. List users who have at least one successful call and at least one failed
--     or errored call, using EXISTS subqueries.

-- 18. Return models that appear in llm_calls but do not appear in model_pricing,
--     using a set operation rather than a join.

-- 19. Produce one distinct list of all locale values from sessions and rubric
--     values from evaluation, using UNION.

-- 20. Return the session_ids that have calls with status 'success' but no calls
--     with status 'timeout', using INTERSECT and EXCEPT.

-- 21. Find call_ids that have an evaluation but no evaluation with verdict
--     'PASS', treating verdict comparison case-insensitively.

-- 22. For each model, report COUNT(logged_cost_usd), COUNT(*), and AVG(logged_cost_usd).
--     Explain what the differences reveal about NULL logged costs.

-- 23. For every session, calculate the average logged_cost_usd and the sum of
--     logged_cost_usd, preserving sessions whose calls all have NULL costs.

-- 24. Compare COUNT(completion_tokens), COUNT(NULLIF(completion_tokens, 0)),
--     and AVG(NULLIF(completion_tokens, 0)) by status. Explain how NULLIF
--     changes the aggregate results.

-- 25. Build a per-user report with total calls, successful calls, evaluated calls,
--     average evaluation score, and total logged cost. Include users with no
--     calls, avoid turning NULL totals into misleading zeroes, and identify the
--     users whose average score is above the average score across evaluated calls.