# GitHub Copilot Configuration Guide

This project includes special configuration files to optimize GitHub Copilot's behavior and prevent VS Code crashes.

## Configuration Files

### 1. `.github/copilot-instructions.md`
**Purpose**: Instructs Copilot on how to generate responses safely

**Key Rules**:
- Maximum 500 lines per response
- Break large summaries into parts
- Use collapsible sections
- Progressive disclosure
- Reference files instead of dumping content

### 2. `.vscode/settings.json`
**Purpose**: VS Code workspace settings for Copilot

**Features**:
- Copilot code generation instructions
- File watcher exclusions (data/, runs/, .cache/)
- Search exclusions for large directories
- Editor optimizations

### 3. `.copilotignore`
**Purpose**: Prevents Copilot from indexing large files

**Excludes**:
- Data directories (data/, runs/)
- Virtual environments (venv/)
- Compiled files (__pycache__/)
- Large model files (*.pt, *.pth)
- Log files (*.log)

## How to Use

### Quick Status Checks
When you ask for status, you'll get a concise response (< 50 lines):

```
User: "check status"

Copilot: 
📊 Quick Status
- ✅ Dataset: 1M+ images ready
- 🔄 Training: 45/100 epochs (ETA: 2 hours)
- 📈 Current mAP: 0.58

Want details? Ask:
- "expand training" - Training metrics and logs
- "expand dataset" - Dataset statistics
```

### Multi-Part Summaries
For large summaries, Copilot will break them into parts:

```
User: "summarize project"

Copilot:
📋 **Project Summary - Part 1 of 3**

[Overview content - max 200 lines]

---
Type "continue" for Part 2: Technical Implementation
```

### Expandable Details
Start with high-level, expand as needed:

```
User: "analyze code"

Copilot:
�� Code Structure Overview
- 5 main modules
- 2000+ lines total
- 33 test cases

Expand: "show src/training" for training module details
```

## Response Size Guidelines

| Request Type | Max Lines | Format |
|--------------|-----------|--------|
| Quick answer | 100 | Direct response |
| Status check | 50 | Bullet points |
| Standard summary | 250 | Sections |
| Detailed analysis | 500 | Multi-part |

## Best Practices

### ✅ DO
- Ask "status" for quick updates
- Use "expand [topic]" for details
- Request "continue" for multi-part responses
- Use collapsible sections for long content

### ❌ DON'T
- Ask for complete file dumps
- Request full log analysis in one go
- Generate deeply nested summaries
- Skip the iterative approach

## Troubleshooting

### VS Code Still Crashing?

1. **Reload Window**: Cmd/Ctrl + Shift + P → "Reload Window"
2. **Clear Copilot Cache**: Delete `~/.vscode/extensions/github.copilot-*`
3. **Reduce Response Size**: Add to custom instructions:
   ```
   Keep all responses under 200 lines maximum
   ```

### Copilot Not Following Rules?

1. Check `.github/copilot-instructions.md` exists
2. Verify `.vscode/settings.json` is valid JSON
3. Reload VS Code window
4. Try explicit instructions: "Please summarize in small parts"

## Commands Reference

| Command | Result |
|---------|--------|
| `status` | Quick status (< 50 lines) |
| `detailed status` | Sectioned with collapsible details |
| `continue` | Next part of multi-part response |
| `expand [topic]` | Detailed view of specific topic |
| `summary` | Multi-part project summary |

## Performance Optimizations

The configuration also includes:

- **File Watcher Exclusions**: Prevents VS Code from watching large directories
- **Search Exclusions**: Faster search by skipping data/runs
- **Large File Optimizations**: Better handling of big files
- **Max Tokenization Length**: Prevents tokenization of very long lines

## Custom Instructions

Add to your user-level Copilot settings for project-wide rules:

1. Open Settings (Cmd/Ctrl + ,)
2. Search "Copilot"
3. Add custom instructions:
   ```
   Always reference .github/copilot-instructions.md
   Generate responses in small, iterative chunks
   Use collapsible sections for large content
   ```

## Examples

### Good Response Pattern
```markdown
# 📊 Training Status

## Current Progress
- Epoch: 45/100
- mAP: 0.58
- Loss: 2.34

## ETA
~2 hours

---
💡 Expand: "show losses" | "show metrics" | "show logs"
```

### Bad Response Pattern (Avoided)
```
[Dumps 2000 lines of training logs]
[Shows all 100 epoch details]
[Pastes entire training script]
[VS Code crashes]
```

## Updates

This configuration will evolve based on:
- VS Code updates
- Copilot API changes
- Project needs
- Performance issues

Check this file periodically for updates.

---

**Last Updated**: November 6, 2025  
**Project**: Robust Thermal Image Object Detection  
**Copilot Version**: GitHub Copilot Chat v0.x
