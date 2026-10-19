# Domain logic library, not an agent plugin

Status: `accepted`

dsh4SCI is a library of materials-science domain logic — the rules for building and judging atomistic structures. It is not an agent framework, an MCP service, or a storage layer. MatPilot (or any other agent) is a **consumer**: it supplies job orchestration, tool registration, permission gates, and persistence; this repository supplies pure functions over structures.

This boundary was chosen because the value here is the physics, not the plumbing. Wrapping it in an agent surface first would have forced every modelling decision to answer "how does MCP see this?" before anyone had settled what the physics even is.

## Considered options

- **MCP service with a TypeScript contract layer.** Rejected: it recreates the cross-process boundary that no domain logic here requires, and inherits a whole class of problems (process isolation, environment whitelisting, port discipline, wakeup callbacks) that exist only to serve that boundary.
- **A MatPilot extension package reusing `candidate_id` addressing and `ToolErrorV1`.** Rejected: it couples the domain rules to one consumer's storage and error vocabulary, so a second consumer pays for decisions made for the first.

## Consequences

- No MCP registration, no HTTP surface, no background task machinery in this repository. Consumers own all of it.
- Exceptions are raised as Python exceptions, not marshalled into an error-code envelope. A consumer that needs `ToolErrorV1` translates at its own boundary.
- Agent ergonomics are delivered as **usage guidance** (a skill and docstrings), not as protocol shape — the library stays importable by anything that speaks Python.
