# Private Solution Infrastructure

This directory contains complete assignment solutions and is **NEVER synced to the public repository**.

## Directory Structure

```
private/
├── solutions/                      # Complete solution implementations
│   ├── backbone_solution.py       # Complete scratch-1 backbone implementation
│   ├── generate_data_solution.py  # Enhanced data generation
│   └── checkpoints/               # Trained model weights for gold standards
│       └── scratch1_gold.pt       # Gold standard checkpoint
└── README.md                       # This file
```

## Solution File Naming Convention

Solution files follow the pattern: `<filename>_solution.py`

This maps to the corresponding student file in `src/assignments/`:
- `backbone_solution.py` → `src/assignments/scratch-1/backbone.py`
- `generate_data_solution.py` → `src/assignments/scratch-1/generate_data.py`

## Managing Solutions

### Inject Solutions (for testing)

```bash
# Inject solutions into assignment directory
python scripts/manage_solutions.py --inject scratch-1

# Now solution code is in src/assignments/scratch-1/
# Run internal tests
pytest tests/internal/test_scratch1_rigor.py -v
```

### Reset to Starter Code

```bash
# Restore original starter code
python scripts/manage_solutions.py --reset scratch-1
```

### List Available Solutions

```bash
python scripts/manage_solutions.py --list
```

## Creating Solutions

1. **Implement the complete solution** in `private/solutions/`:
   ```bash
   # Create solution file
   touch private/solutions/backbone_solution.py

   # Implement complete solution (no TODOs)
   # Include full implementation with DINOv2, projector, etc.
   ```

2. **Train and save gold standard**:
   ```bash
   # Inject solution for training
   python scripts/manage_solutions.py --inject scratch-1

   # Train the model
   cd src/assignments/scratch-1
   python backbone.py

   # Save best checkpoint as gold standard
   cp checkpoints/best_model.pt ../../../tests/internal/fixtures/scratch1_gold.pt
   ```

3. **Write internal tests** in `tests/internal/`:
   - Gradient leak detection
   - Latent fidelity comparison
   - Training convergence
   - Edge cases

4. **Add solution hints** to student starter code:
   ```python
   # In src/assignments/scratch-1/backbone.py
   def forward(self, x):
       # TODO: [SOLUTION] Use torch.rsqrt for numerical stability
       pass
   ```

   These hints will be automatically removed during sanitization.

## Workflow for Releasing Assignments

1. **Develop in private repo**:
   - Create complete solution in `private/solutions/`
   - Write internal tests in `tests/internal/`
   - Add `TODO: [SOLUTION]` hints in starter code

2. **Test thoroughly**:
   ```bash
   # Test with solutions injected
   python scripts/manage_solutions.py --inject scratch-1
   pytest tests/internal/test_scratch1_rigor.py -v

   # Reset and test starter code
   python scripts/manage_solutions.py --reset scratch-1
   pytest tests/public/test_scratch1_basic.py -v
   ```

3. **Create release tag**:
   ```bash
   # Tag triggers automatic sanitization and sync
   git tag release-scratch-1
   git push origin release-scratch-1
   ```

4. **Review and merge**:
   - GitHub Actions automatically sanitizes and pushes to public repo
   - Review the `public-release-*` branch
   - Merge to main/staging after review

## Security Notes

⚠️ **IMPORTANT**: This directory contains complete solutions and must NEVER be pushed to the public repository.

- `.gitignore` excludes `private/` from commits on public branches
- Sanitization script deletes this directory before public sync
- Double-check that no solution code leaks into public commits

## Gold Standard Generation

Gold standards are reference outputs used for rigorous testing:

```python
# In tests/internal/fixtures/
scratch1_gold_output.pt    # Expected projector output
scratch1_gold_checkpoint.pt # Trained model weights
```

To generate:

```bash
# Inject solution
python scripts/manage_solutions.py --inject scratch-1

# Run training and save checkpoints
cd src/assignments/scratch-1
python backbone.py

# Copy best checkpoint
cp checkpoints/best_model.pt ../../../tests/internal/fixtures/scratch1_gold.pt
```

## Troubleshooting

### Solutions not injecting
- Check that solution files follow naming convention: `*_solution.py`
- Verify target directory exists: `src/assignments/scratch-1/`
- Check file permissions

### Tests failing after injection
- Verify solution code is complete and correct
- Check import paths in test files
- Ensure gold standard files exist in `tests/internal/fixtures/`

### Accidental leak to public repo
- Immediately revert the commit
- Run sanitization manually: `bash scripts/sanitize.sh`
- Force push to public repo with clean state
