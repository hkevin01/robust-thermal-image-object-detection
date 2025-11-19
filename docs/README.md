# 📚 Documentation Directory

Welcome to the documentation directory for the Robust Thermal Image Object Detection project.

## 📖 Documentation Structure

### 🎯 Active Documentation

These are the current, maintained documentation files:

| Document | Purpose | Audience |
|----------|---------|----------|
| [Competition Submission Guide](COMPETITION_SUBMISSION_GUIDE.md) | Detailed CodaLab submission instructions | All users |
| [Submission Workflow](SUBMISSION_WORKFLOW.md) | Quick-start submission guide | Quick reference |
| [Submission Checklist](SUBMISSION_CHECKLIST.md) | Phase-by-phase checklist | Step-by-step |
| [Memory Bank](MEMORY_BANK.md) | Competition knowledge base | Reference |
| [Quick Reference](QUICK_REFERENCE.md) | Common commands cheat sheet | Quick reference |

### 📜 Historical Documentation

Historical status files documenting the development journey are indexed in [STATUS_FILES_INDEX.md](STATUS_FILES_INDEX.md).

## 🗂️ File Organization Guidelines

### ✅ DO Create Here:
- **Guides & Tutorials**: Step-by-step instructions
- **API Documentation**: Code documentation
- **Design Documents**: Architecture decisions
- **Troubleshooting Guides**: Problem-solving resources
- **Historical Records**: Status updates, debugging logs
- **Reference Materials**: Cheat sheets, command lists

### ❌ DON'T Create in Root:
- Markdown files (except README.md, LICENSE)
- Status update files
- Temporary documentation
- Debug logs in markdown format

**Exception**: The main `README.md` stays in the project root as the entry point.

## 📝 Creating New Documentation

When creating new documentation:

1. **Choose the right location**:
   - Tutorials/Guides → `docs/guides/`
   - API docs → `docs/api/`
   - Status updates → Add to Changelog in main README.md
   - Troubleshooting → `docs/troubleshooting/`

2. **Use clear naming**:
   - UPPERCASE_WITH_UNDERSCORES.md for major documents
   - lowercase-with-dashes.md for specific guides
   - Descriptive names (not generic like STATUS.md)

3. **Link from appropriate places**:
   - Update this README if it's a major document
   - Link from main README.md if it's important for new users
   - Cross-reference related documents

4. **Keep it updated**:
   - Mark outdated docs as [ARCHIVED] or [HISTORICAL]
   - Update the modification date
   - Remove or archive obsolete content

## 🔗 Quick Links

### For Users
- **Getting Started**: See main [README.md](../README.md)
- **Installation**: See [README.md#installation](../README.md#-installation)
- **Usage**: See [README.md#usage](../README.md#-usage)

### For Developers
- **Contributing**: See [README.md#contributing](../README.md#-contributing)
- **Testing**: See [README.md#testing](../README.md#-testing)
- **Development Setup**: See [README.md#development-setup](../README.md#development-setup)

### For Competition Submission
- **Submission Guide**: [COMPETITION_SUBMISSION_GUIDE.md](COMPETITION_SUBMISSION_GUIDE.md)
- **Quick Workflow**: [SUBMISSION_WORKFLOW.md](SUBMISSION_WORKFLOW.md)
- **Checklist**: [SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md)

## 📊 Documentation Standards

### Markdown Formatting
- Use `#` for titles, `##` for sections, `###` for subsections
- Include table of contents for documents >500 lines
- Use code blocks with language specification: ```python
- Include emoji sparingly for visual organization (📚 🎯 ✅ ❌ etc.)

### Code Examples
- Always test code examples before including
- Provide context (what does this do?)
- Show expected output when relevant
- Include error handling examples

### Linking
- Use relative links for internal documentation
- Use absolute links for external resources
- Always check links work after creation
- Keep link text descriptive

## 🔄 Maintenance

This directory is maintained as part of the project. To keep it clean:

1. **Regular Audits**: Review docs quarterly
2. **Archive Old Content**: Move outdated docs to archive subdirectories
3. **Update Index**: Keep this README current
4. **Remove Duplicates**: Consolidate overlapping content

## 📞 Questions?

For questions about documentation:
- Check existing docs first
- Review the main [README.md](../README.md)
- Open an issue with the `documentation` label

---

**Last Updated**: November 18, 2025
