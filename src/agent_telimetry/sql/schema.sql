CREATE TABLE model_pricing (
	model VARCHAR(30) PRIMARY KEY,
	prompt_usd_per_token Decimal,
	completion_usd_per_token Decimal(1, 30),
	effective_from Date
);

CREATE TABLE sessions (
	session_id Text PRIMARY KEY,
	user_id TEXT NOT NULL,
	app_version TEXT NOT NULL,
	device text NOT NULL,
	locale varchar(10) NOT NULL,
	plan varchar(10) NOT NULL,
	started_at timestamp NOT NULL,
	network varchar(15) NOT NULL
);

CREATE TABLE llm_calls (
	call_id varchar(15) PRIMARY KEY,
	session_id varchar(15) NOT NULL,
	idempotency_key varchar(15) NOT NULL,
	model varchar(30) NOT NULL,
	tool_name varchar(30) NOT NULL,
	prompt_tokens int NOT NULL,
	completion_tokens int NOT NULL,
	cache_hit bool NOT NULL,
	latency_ms int NOT NULL,
	status varchar(20) NOT NULL,
	logged_cost_usd decimal(1,30),
	called_at timestamp NOT NULL,
	attempt_no int NOT NULL,
	FOREIGN KEY(model) REFERENCES model_pricing(model)
);

ALTER TABLE llm_calls
	ADD CONSTRAINT fk_model_session FOREIGN KEY (session_id) REFERENCES sessions(session_id);

CREATE TABLE evaluation (
	eval_id varchar(12) PRIMARY KEY,
	call_id varchar(15) NOT NULL REFERENCES llm_calls(call_id),
	rubric varchar(30) NOT NULL,
	verdict varchar(5) NOT NULL,
	score decimal(1,4) NOT NULL,
	evaluator varchar(20) NOT NULL,
	evaluated_at timestamp NOT NULL
);

ALTER TABLE llm_calls
	ADD CONSTRAINT chk_min_lenght_0 CHECK (
		prompt_tokens >= 0 AND
		completion_tokens >= 0
	)



