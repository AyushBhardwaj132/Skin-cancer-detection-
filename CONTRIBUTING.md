# Contributing to ISIC 2024 Skin Cancer Detection & Diagnostics System

First off, thank you for considering contributing to this open-source medical AI initiative!

---

## 📜 Development Guidelines

### 1. Code Style & Standards
- Follow **PEP 8** guidelines.
- Use explicit type annotations (`from __future__ import annotations`).
- Ensure all public classes, functions, and modules include Google-style docstrings.

### 2. Testing & Quality Checks
Before submitting a Pull Request, verify that all unit and integration tests pass:

```bash
# Run pytest test suite
python -m pytest tests/

# Run 20-step complete verification health check suite
python src/verify_and_run.py
```

### 3. Pull Request Process
1. Fork the repository and create a new feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Commit your changes with structured commit messages (`feat: ...`, `fix: ...`, `docs: ...`, `perf: ...`).
3. Ensure no regressions occur in the 20-step verification suite.
4. Open a Pull Request targeting the `main` or `production-refactor` branch.

---

## 🔒 Security & Medical Disclaimer Notice
This repository is strictly for research, benchmarking, and educational purposes. Ensure no confidential patient health information (PHI) or un-anonymized medical images are committed to the repository.
