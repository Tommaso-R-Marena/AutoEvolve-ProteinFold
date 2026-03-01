#!/usr/bin/env python3
"""Expand protein database by fetching from UniProt and AlphaFold."""
import sqlite3
import argparse
import requests
import time
import json
from pathlib import Path
from typing import List, Dict
import hashlib

class ProteinDatabase:
    """Manage expanding protein sequence and structure database."""
    
    def __init__(self, db_path: str = 'data/protein_database/proteins.db'):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = None
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database with schema."""
        self.conn = sqlite3.connect(str(self.db_path))
        cursor = self.conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS proteins (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uniprot_id TEXT UNIQUE,
                sequence TEXT NOT NULL,
                sequence_hash TEXT UNIQUE,
                length INTEGER,
                description TEXT,
                organism TEXT,
                has_structure BOOLEAN DEFAULT 0,
                structure_data BLOB,
                added_timestamp REAL,
                last_used_timestamp REAL
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_sequence_hash ON proteins(sequence_hash)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_length ON proteins(length)
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fetch_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fetch_timestamp REAL,
                source TEXT,
                sequences_added INTEGER,
                structures_added INTEGER
            )
        ''')
        
        self.conn.commit()
    
    def get_database_size_mb(self) -> float:
        """Get current database size in MB."""
        if self.db_path.exists():
            return self.db_path.stat().st_size / (1024 * 1024)
        return 0.0
    
    def get_protein_count(self) -> int:
        """Get total number of proteins in database."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM proteins')
        return cursor.fetchone()[0]
    
    def add_protein(self, uniprot_id: str, sequence: str, description: str = '', 
                   organism: str = '', structure_data: bytes = None) -> bool:
        """Add protein to database if not already present."""
        sequence_hash = hashlib.md5(sequence.encode()).hexdigest()
        
        cursor = self.conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO proteins 
                (uniprot_id, sequence, sequence_hash, length, description, organism, 
                 has_structure, structure_data, added_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                uniprot_id,
                sequence,
                sequence_hash,
                len(sequence),
                description,
                organism,
                structure_data is not None,
                structure_data,
                time.time()
            ))
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            # Already exists
            return False
    
    def fetch_from_uniprot(self, n_samples: int = 100) -> int:
        """Fetch proteins from UniProt and add to database."""
        added_count = 0
        
        try:
            # Random sampling from reviewed proteins
            url = f"https://rest.uniprot.org/uniprotkb/stream?format=json&query=reviewed:true&size={n_samples}"
            
            print(f"Fetching {n_samples} proteins from UniProt...")
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            results = data.get('results', [])
            
            for entry in results:
                uniprot_id = entry.get('primaryAccession', '')
                sequence = entry.get('sequence', {}).get('value', '')
                
                if not sequence:
                    continue
                
                description = entry.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', '')
                organism = entry.get('organism', {}).get('scientificName', '')
                
                if self.add_protein(uniprot_id, sequence, description, organism):
                    added_count += 1
            
            print(f"✓ Added {added_count} new proteins from UniProt")
            
        except Exception as e:
            print(f"Error fetching from UniProt: {e}")
        
        return added_count
    
    def fetch_alphafold_structure(self, uniprot_id: str) -> bytes:
        """Fetch structure from AlphaFold DB."""
        try:
            url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                return response.content
        except:
            pass
        
        return None
    
    def enrich_with_structures(self, max_structures: int = 20):
        """Add AlphaFold structures to proteins without structures."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT uniprot_id FROM proteins 
            WHERE has_structure = 0 
            LIMIT ?
        ''', (max_structures,))
        
        proteins_without_structures = cursor.fetchall()
        added_count = 0
        
        print(f"Attempting to fetch structures for {len(proteins_without_structures)} proteins...")
        
        for (uniprot_id,) in proteins_without_structures:
            structure_data = self.fetch_alphafold_structure(uniprot_id)
            
            if structure_data:
                cursor.execute('''
                    UPDATE proteins 
                    SET has_structure = 1, structure_data = ?
                    WHERE uniprot_id = ?
                ''', (structure_data, uniprot_id))
                added_count += 1
                
            time.sleep(0.5)  # Rate limiting
        
        self.conn.commit()
        print(f"✓ Added {added_count} structures")
        
        return added_count
    
    def log_fetch(self, source: str, sequences_added: int, structures_added: int):
        """Log fetch operation."""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO fetch_history (fetch_timestamp, source, sequences_added, structures_added)
            VALUES (?, ?, ?, ?)
        ''', (time.time(), source, sequences_added, structures_added))
        self.conn.commit()
    
    def get_batch_for_training(self, batch_size: int = 32) -> List[Dict]:
        """Get batch of proteins for training, prioritizing those with structures."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT uniprot_id, sequence, structure_data, has_structure
            FROM proteins
            WHERE has_structure = 1
            ORDER BY RANDOM()
            LIMIT ?
        ''', (batch_size,))
        
        results = cursor.fetchall()
        
        batch = []
        for uniprot_id, sequence, structure_data, has_structure in results:
            batch.append({
                'uniprot_id': uniprot_id,
                'sequence': sequence,
                'structure_data': structure_data,
                'has_structure': bool(has_structure)
            })
        
        # Update last used timestamp
        for item in batch:
            cursor.execute('''
                UPDATE proteins SET last_used_timestamp = ? WHERE uniprot_id = ?
            ''', (time.time(), item['uniprot_id']))
        
        self.conn.commit()
        return batch
    
    def get_stats(self) -> Dict:
        """Get database statistics."""
        cursor = self.conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM proteins')
        total_proteins = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM proteins WHERE has_structure = 1')
        proteins_with_structures = cursor.fetchone()[0]
        
        cursor.execute('SELECT AVG(length) FROM proteins')
        avg_length = cursor.fetchone()[0] or 0
        
        return {
            'total_proteins': total_proteins,
            'proteins_with_structures': proteins_with_structures,
            'avg_sequence_length': round(avg_length, 1),
            'database_size_mb': round(self.get_database_size_mb(), 2)
        }
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()

def main():
    parser = argparse.ArgumentParser(description='Expand protein database')
    parser.add_argument('--max-size-gb', type=float, default=2.0,
                       help='Maximum database size in GB')
    parser.add_argument('--fetch-sequences', type=int, default=100,
                       help='Number of sequences to fetch per run')
    parser.add_argument('--fetch-structures', type=int, default=20,
                       help='Number of structures to fetch per run')
    
    args = parser.parse_args()
    
    db = ProteinDatabase()
    
    # Check current size
    current_size_mb = db.get_database_size_mb()
    max_size_mb = args.max_size_gb * 1024
    
    print(f"\nCurrent database size: {current_size_mb:.2f} MB")
    print(f"Maximum allowed size: {max_size_mb:.2f} MB ({args.max_size_gb} GB)")
    print(f"Available space: {max_size_mb - current_size_mb:.2f} MB\n")
    
    if current_size_mb >= max_size_mb:
        print("⚠️  Database at maximum size. Not fetching more data.")
        stats = db.get_stats()
        print(f"\nDatabase Stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        db.close()
        return
    
    # Fetch new sequences
    print("Expanding database...\n")
    
    sequences_added = db.fetch_from_uniprot(args.fetch_sequences)
    
    # Check size again
    if db.get_database_size_mb() < max_size_mb:
        structures_added = db.enrich_with_structures(args.fetch_structures)
    else:
        structures_added = 0
        print("⚠️  Size limit reached, skipping structure fetch")
    
    # Log this fetch
    db.log_fetch('uniprot+alphafold', sequences_added, structures_added)
    
    # Print stats
    stats = db.get_stats()
    print(f"\n{'='*50}")
    print("Database Statistics:")
    print(f"{'='*50}")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print(f"{'='*50}\n")
    
    db.close()
    
    print("✅ Database expansion complete!")

if __name__ == '__main__':
    main()
