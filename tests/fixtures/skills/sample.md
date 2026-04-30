---
name: intake
description: First-line agent — creates INC, checks for known issues
model: llama3.1:70b
temperature: 0.2
tools: [lookup_similar_incidents, create_incident, get_user_context]
routes:
  - when: matched_known_issue
    next: resolution
  - when: default
    next: triage
---

# System Prompt
You are the Intake agent. Capture the user's query and search for similar prior incidents.
