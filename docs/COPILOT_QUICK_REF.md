# 🤖 Copilot Quick Reference

## ✅ Configuration Complete

Three files created to prevent VS Code crashes:

1. **`.github/copilot-instructions.md`** - Response size rules
2. **`.vscode/settings.json`** - Workspace configuration  
3. **`.copilotignore`** - Exclude large directories

## 📏 Response Size Limits

| Type | Max Lines |
|------|-----------|
| Status | 50 |
| Quick answer | 100 |
| Summary | 250 |
| Detailed | 500 (multi-part) |

## 🎯 How to Ask

### Good ✅
```
"status"                    → Quick update
"expand dataset"            → Specific details
"continue"                  → Next part
"summarize in parts"        → Safe summary
```

### Bad ❌
```
"show everything"           → Too large
"dump all logs"             → Crashes VS Code
"full project analysis"     → Use parts instead
```

## 🔧 Commands

- `status` - Quick status (< 50 lines)
- `expand [topic]` - Details on specific topic
- `continue` - Next section of multi-part response
- `summary` - Project summary in parts

## 💡 Tips

1. **Start small**: Ask "status" first
2. **Expand as needed**: Use "expand X" for details
3. **Multi-part**: Large responses come in parts
4. **Reference files**: Copilot shows paths, not full files

## 🚨 If VS Code Crashes

1. Reload: `Ctrl/Cmd + Shift + P` → "Reload Window"
2. Ask in smaller chunks: "show only training status"
3. Check: `.github/copilot-instructions.md` exists

## 📚 Full Guide

See `.github/COPILOT_GUIDE.md` for complete documentation.

---

**Remember**: Small chunks = Happy VS Code! 🚀
