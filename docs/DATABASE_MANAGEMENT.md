# Protein Database Management

## Overview

The model maintains an ever-expanding SQLite database of protein sequences and structures that grows with each training cycle.

## Database Structure

### Location
`data/protein_database/proteins.db`

### Schema

**proteins table:**
- `uniprot_id`: UniProt accession ID
- `sequence`: Amino acid sequence
- `sequence_hash`: MD5 hash for deduplication
- `length`: Sequence length
- `description`: Protein name/description
- `organism`: Source organism
- `has_structure`: Boolean indicating if 3D structure available
- `structure_data`: PDB file data (BLOB)
- `added_timestamp`: When added
- `last_used_timestamp`: Last used in training

**fetch_history table:**
- Tracks all data fetch operations
- Source, timestamp, counts

## Size Management

### Git LFS Integration

The database uses **Git Large File Storage (LFS)** to handle large files efficiently:

```bash
# Files tracked by LFS (see .gitattributes)
*.db
*.sqlite
*.pt (model weights)
*.h5
```

### Size Limits

**Default configuration:**
- Maximum database size: **2 GB** (configurable)
- Each training cycle can add: ~100 sequences + 20 structures
- Estimated growth: ~10-50 MB per cycle
- At 2GB limit: ~50,000-100,000 protein sequences

**GitHub LFS limits (free tier):**
- Storage: 1 GB included
- Bandwidth: 1 GB/month included
- Additional packs available if needed

### Configuration

Adjust in workflow or manually:

```bash
python scripts/expand_protein_database.py --max-size-gb 2.0
```

## Data Sources

### UniProt
- **Source**: https://rest.uniprot.org/
- **Type**: Protein sequences + metadata
- **Fetch rate**: 100 sequences per cycle (configurable)
- **No API key required**

### AlphaFold Database
- **Source**: https://alphafold.ebi.ac.uk/
- **Type**: Predicted 3D structures (PDB format)
- **Fetch rate**: 20 structures per cycle (configurable)
- **No API key required**

## Usage in Training

### Batch Retrieval

```python
from scripts.expand_protein_database import ProteinDatabase

db = ProteinDatabase()
batch = db.get_batch_for_training(batch_size=32)

for protein in batch:
    sequence = protein['sequence']
    structure = protein['structure_data']
    # Train on real data
```

### Statistics

```bash
python scripts/expand_protein_database.py
```

Outputs:
- Total proteins
- Proteins with structures
- Average sequence length
- Database size (MB)

## Deduplication

Sequences are deduplicated using MD5 hashes:
- Identical sequences are not added twice
- UniProt ID uniqueness is enforced
- Structure data updates existing entries

## Performance Optimization

### Indexes
- `sequence_hash`: Fast duplicate detection
- `length`: Batch selection by size

### Query Optimization
- Prioritizes proteins with structures for training
- Random sampling for diversity
- Tracks usage to balance dataset

## Backup & Portability

The database is automatically:
1. **Committed** to Git (via LFS) after each expansion
2. **Versioned** with the rest of the repository
3. **Transferable** to other machines via `git lfs pull`

## Expansion Strategy

**Every 6-hour training cycle:**
1. Check current database size vs limit
2. If space available:
   - Fetch 100 new sequences from UniProt
   - Fetch 20 new structures from AlphaFold
   - Deduplicate and store
3. Use database for training (mixed with synthetic data)
4. Commit updated database

**Growth trajectory:**
- Week 1: ~1,000 proteins
- Month 1: ~15,000 proteins
- Month 3: ~45,000 proteins
- Month 6: ~90,000 proteins (approaching 2GB limit)

## When Database is Full

Options when reaching size limit:

1. **Increase limit** (if GitHub LFS quota allows)
2. **Prune old/unused entries**
3. **Compress structures** (store only coordinates, not full PDB)
4. **External storage** (database on S3/cloud, fetch as needed)

Current implementation: **Stops fetching when limit reached**, continues training on existing data.

## Cost Considerations

**Free tier (current):**
- 1 GB storage + 1 GB bandwidth/month
- Sufficient for 2GB database if commit frequency is managed
- LFS only transfers on pull, not on clone

**If exceeded:**
- Data packs available: $5/month for 50 GB storage + 50 GB bandwidth
- Or migrate to self-hosted Git LFS server

## Monitoring

Database stats are logged after each expansion:

```
==================================================
Database Statistics:
==================================================
  total_proteins: 5432
  proteins_with_structures: 1876
  avg_sequence_length: 287.3
  database_size_mb: 245.67
==================================================
```

Tracked in `fetch_history` table for analysis.
