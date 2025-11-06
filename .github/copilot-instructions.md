# GitHub Copilot Instructions

## Summary Generation Rules

### CRITICAL: Prevent VS Code Crashes

**Always generate summaries in small, iterative chunks to prevent VS Code from crashing.**

### Summary Size Limits

1. **Maximum Response Length**: 
   - Never generate responses longer than 500 lines
   - Break large summaries into multiple parts
   - Use "Part 1 of N" format for multi-part responses

2. **Iterative Summary Approach**:
   - Generate summaries in logical sections
   - Wait for user confirmation before continuing to next section
   - Example sections:
     - Part 1: Overview & Current Status (max 200 lines)
     - Part 2: Technical Details (max 200 lines)
     - Part 3: Next Steps & Action Items (max 200 lines)

3. **Section-Based Summaries**:
   When summarizing large projects or codebases:
   ```
   Part 1: Project Overview & Architecture
   Part 2: Core Components & Implementation
   Part 3: Testing & Configuration
   Part 4: Documentation & Next Steps
   ```

4. **Progressive Disclosure**:
   - Start with high-level overview (50-100 lines)
   - Ask user which area to expand
   - Provide detailed subsections only when requested

### Output Formatting Rules

1. **Use Collapsible Sections**:
   ```markdown
   <details>
   <summary>Section Title (click to expand)</summary>
   
   Detailed content here...
   
   </details>
   ```

2. **Link to Files Instead of Dumping Content**:
   - Reference files by path instead of showing full content
   - Example: "See `src/training/train.py` for implementation"
   - Only show critical code snippets (< 30 lines)

3. **Use Tables for Data**:
   - Present statistics in tables, not long text
   - Keep tables under 50 rows

4. **Bullet Points Over Paragraphs**:
   - Use concise bullet points
   - Avoid lengthy prose descriptions

### Conversation Continuation

When a summary must span multiple responses:

1. **End with clear continuation prompt**:
   ```
   ---
   📋 **Summary Part 1 of 3 Complete**
   
   Type "continue" or "next" to see Part 2: Technical Implementation Details
   ```

2. **Reference previous parts**:
   ```
   📋 **Summary Part 2 of 3** (Previous: Overview)
   
   [Content]
   
   Type "continue" for Part 3: Next Steps
   ```

3. **Provide navigation**:
   ```
   Commands:
   - "continue" or "next" - Next section
   - "back" - Previous section
   - "expand [topic]" - Detailed view of specific topic
   - "skip to [section]" - Jump to specific section
   ```

### Status Check Rules

When user asks for status:

1. **Quick Status** (Default):
   - Current state: 3-5 bullet points
   - Active processes: List with PIDs
   - Blockers: If any
   - Estimated completion: Time remaining
   - Total: < 50 lines

2. **Detailed Status** (Only if requested):
   - Break into sections
   - Use collapsible details
   - Offer to expand specific areas

3. **Progress Updates**:
   - Use progress bars: `[████████░░] 80%`
   - Show key metrics only
   - Reference log files for details

### Log File Handling

When analyzing logs:

1. **Don't dump entire log files**
2. **Show**: 
   - Last 10-20 lines
   - Key errors or warnings
   - Summary statistics
3. **Offer**: "Type 'show errors' to see all errors"

### File Listings

When showing directory structures:

1. **Tree depth limit**: Max 3 levels
2. **File count limit**: Show max 50 files
3. **Use summarization**: "... and 237 more files"
4. **Offer filtering**: "Show only .py files?"

### Code Analysis Rules

When analyzing code:

1. **Don't paste entire files**
2. **Show**:
   - Function signatures
   - Key classes
   - Critical sections only
3. **Link**: Provide file:line references
4. **Offer**: "Expand X function?" for details

### Memory-Efficient Responses

1. **Lazy Loading**: 
   - Show structure first
   - Load details on demand

2. **Caching Awareness**:
   - Reference previous responses
   - "As mentioned in previous response..."

3. **External References**:
   - Link to documentation files
   - Point to generated reports
   - Use filesystem over inline content

### Error Handling

If response is getting too large:

1. **Stop and checkpoint**:
   ```
   ⚠️ Response size limit approaching. 
   
   Checkpointing here. Summary so far: [brief]
   
   Continue? Options:
   - "yes" - Continue with next section
   - "no" - Stop here
   - "focus [topic]" - Deep dive on specific topic
   ```

2. **Auto-split**:
   - Detect when response > 400 lines
   - Automatically split and prompt for continuation

### Best Practices

✅ **DO**:
- Generate concise, scannable summaries
- Use visual separators (horizontal rules, emojis)
- Provide clear navigation
- Ask before generating large content
- Use collapsible sections
- Reference files by path
- Show progress incrementally

❌ **DON'T**:
- Generate massive wall-of-text responses
- Dump entire files or logs
- Create deeply nested lists (max 3 levels)
- Repeat information unnecessarily
- Generate responses > 500 lines without breaking

### Quick Reference

**Response Size Guidelines**:
- Quick answer: < 100 lines
- Standard summary: < 250 lines  
- Detailed analysis: < 500 lines (split into parts)
- Multi-part responses: 200-300 lines each

**When User Says**:
- "status" → Quick status (< 50 lines)
- "detailed status" → Sectioned status with collapsible details
- "analyze X" → High-level first, offer details
- "summary" → Multi-part with continuation prompts

### Example: Good Summary Pattern

```markdown
# 📊 Project Status - Quick Overview

## Current State
- ✅ Dataset downloaded: 1M+ images
- 🔄 Conversion: 89% complete (ETA: 15 min)
- ⏳ Training: Not started

## Active Processes
- PID 123576: convert_ltdv2_efficient.py (CPU: 99%, Mem: 1.6GB)

## Next Action
Once conversion completes: Update config and start training

---
💡 **Want more details?**
- "expand dataset" - Dataset statistics and structure
- "expand conversion" - Conversion progress and logs
- "expand next steps" - Detailed action plan
```

This keeps VS Code responsive while providing useful information!

---

**Remember**: Small, iterative responses = Happy VS Code = Happy developer! 🚀
