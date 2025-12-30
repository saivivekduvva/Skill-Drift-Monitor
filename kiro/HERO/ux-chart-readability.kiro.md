# UX Improvement Spec – Curriculum Relevance Chart

## Problem
Users think the horizontal bar chart shows job counts.

## Evidence
Peer feedback: chart is misinterpreted as number of jobs.

## Root Cause
- X-axis label is technical.
- No visible explanation.
- Score range unclear.
- Tooltip lacks meaning.

## Decision
Refactor chart to improve human readability.

## Planned Changes
1. Add explanation card above chart.
2. Rename x-axis to "Curriculum ↔ Industry Match Score".
3. Force x-axis range to 0–1.
4. Improve hover tooltip.

## Implementation Reference
File: app.py  
Section: Curriculum Relevance Ranking chart

## Success Criteria
A first-time user must understand chart meaning in under 5 seconds.