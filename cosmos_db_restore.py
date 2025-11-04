#!/usr/bin/env python3
"""
Cosmos DB Database Restore Script

Restores containers from JSON backup files to a Cosmos DB database.

Usage:
    python cosmos_db_restore.py --database <db_name> --input-dir <backup_dir>
    python cosmos_db_restore.py --database <db_name> --input-dir <backup_dir> --containers container1 container2
"""

import os
import sys
import json
import argparse
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from azure.cosmos import CosmosClient, exceptions
from azure.identity import DefaultAzureCredential

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class CosmosDBRestore:
    """Handles restoration of Cosmos DB containers from JSON files."""
    
    def __init__(self, cosmos_db_id: str):
        self.cosmos_db_uri = f"https://{cosmos_db_id}.documents.azure.com:443/"
        self.credential = DefaultAzureCredential()
        self.client = None
        
    def connect(self) -> bool:
        """Establish connection to Cosmos DB."""
        try:
            logger.info(f"Connecting to Cosmos DB at {self.cosmos_db_uri}")
            self.client = CosmosClient(
                self.cosmos_db_uri,
                credential=self.credential,
                consistency_level="Session"
            )
            logger.info("Successfully connected to Cosmos DB")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Cosmos DB: {e}")
            return False
    
    def container_exists(self, database_name: str, container_name: str) -> bool:
        """Check if a container exists."""
        try:
            db = self.client.get_database_client(database_name)
            container = db.get_container_client(container_name)
            container.read()
            return True
        except exceptions.CosmosResourceNotFoundError:
            return False
        except Exception as e:
            logger.error(f"Error checking container: {e}")
            return False
    
    def restore_container(
        self,
        database_name: str,
        container_name: str,
        input_file: Path,
        overwrite: bool = False
    ) -> Dict[str, Any]:
        """Restore a single container from a JSON file."""
        stats = {
            'container': container_name,
            'items_restored': 0,
            'items_skipped': 0,
            'failed_items': 0,
            'file_path': str(input_file),
            'errors': []
        }
        
        try:
            # Check if container exists
            if not self.container_exists(database_name, container_name):
                error_msg = f"Container '{container_name}' does not exist in database '{database_name}'. Please create it first."
                logger.error(error_msg)
                stats['errors'].append(error_msg)
                return stats
            
            db = self.client.get_database_client(database_name)
            container = db.get_container_client(container_name)
            
            logger.info(f"Restoring container '{container_name}' from {input_file}")
            
            # Read items from backup file
            with open(input_file, 'r', encoding='utf-8') as f:
                items = json.load(f)
            
            total_items = len(items)
            logger.info(f"  Found {total_items} items in backup file")
            
            # Restore items
            for idx, item in enumerate(items, 1):
                try:
                    item_id = item.get('id', 'unknown')
                    
                    if overwrite:
                        # Upsert (create or replace)
                        container.upsert_item(item)
                        stats['items_restored'] += 1
                    else:
                        # Try to create only (will fail if exists)
                        try:
                            container.create_item(item)
                            stats['items_restored'] += 1
                        except exceptions.CosmosResourceExistsError:
                            stats['items_skipped'] += 1
                            logger.debug(f"  Item {item_id} already exists, skipping")
                    
                    if idx % 100 == 0:
                        logger.info(f"  Progress: {idx}/{total_items} items...")
                        
                except exceptions.CosmosHttpResponseError as e:
                    stats['failed_items'] += 1
                    error_msg = f"Failed to restore item: {str(e)}"
                    stats['errors'].append(error_msg)
                    logger.error(f"  {error_msg}")
                    
                except Exception as e:
                    stats['failed_items'] += 1
                    error_msg = f"Unexpected error: {str(e)}"
                    stats['errors'].append(error_msg)
                    logger.error(f"  {error_msg}")
            
            logger.info(f"✓ Restored {stats['items_restored']} items to '{container_name}'")
            if stats['items_skipped'] > 0:
                logger.info(f"  Skipped {stats['items_skipped']} existing items")
            
        except FileNotFoundError:
            error_msg = f"Backup file not found: {input_file}"
            logger.error(error_msg)
            stats['errors'].append(error_msg)
            
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in backup file: {str(e)}"
            logger.error(error_msg)
            stats['errors'].append(error_msg)
            
        except Exception as e:
            error_msg = f"Critical error restoring container '{container_name}': {str(e)}"
            logger.error(error_msg)
            stats['errors'].append(error_msg)
        
        return stats
    
    def restore_database(
        self,
        database_name: str,
        input_dir: Path,
        containers: Optional[List[str]] = None,
        overwrite: bool = False
    ) -> Dict[str, Any]:
        """Restore entire database or specific containers from backup."""
        restore_stats = {
            'database': database_name,
            'containers': {},
            'total_restored': 0,
            'total_skipped': 0,
            'total_failures': 0
        }
        
        # Get list of backup files
        if containers:
            backup_files = [(c, input_dir / f"{c}.json") for c in containers]
        else:
            backup_files = [
                (f.stem, f) for f in input_dir.glob("*.json") 
                if not f.name.startswith('_')
            ]
        
        if not backup_files:
            logger.error(f"No backup files found in {input_dir}")
            return restore_stats
        
        logger.info(f"Found {len(backup_files)} containers to restore")
        
        # Restore each container
        for container_name, backup_file in backup_files:
            if not backup_file.exists():
                logger.error(f"Backup file not found: {backup_file}")
                continue
                
            stats = self.restore_container(
                database_name,
                container_name,
                backup_file,
                overwrite
            )
            
            restore_stats['containers'][container_name] = stats
            restore_stats['total_restored'] += stats['items_restored']
            restore_stats['total_skipped'] += stats['items_skipped']
            restore_stats['total_failures'] += stats['failed_items']
        
        return restore_stats


def confirm_restore(
    database_name: str,
    num_containers: int,
    overwrite: bool
) -> bool:
    """Ask user to confirm the restore operation."""
    print("\n" + "="*70)
    print("RESTORE CONFIRMATION")
    print("="*70)
    print(f"Database:    {database_name}")
    print(f"Containers:  {num_containers}")
    print(f"Mode:        {'OVERWRITE (will replace existing items)' if overwrite else 'CREATE ONLY (skip existing items)'}")
    print("\nWARNING: This operation will restore data to the database.")
    if overwrite:
        print("Existing items with the same ID will be OVERWRITTEN!")
    print("="*70)
    
    response = input("\nDo you want to proceed? (yes/no): ").strip().lower()
    return response in ['yes', 'y']


def main():
    parser = argparse.ArgumentParser(
        description='Restore Cosmos DB database from JSON backup files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Restore entire database (create only, skip existing)
  python cosmos_db_restore.py --database mydb --input-dir ./backups/mydb_backup
  
  # Restore specific containers
  python cosmos_db_restore.py --database mydb --input-dir ./backups --containers conversations schedules
  
  # Restore with overwrite mode
  python cosmos_db_restore.py --database mydb --input-dir ./backups --overwrite
  
  # Skip confirmation
  python cosmos_db_restore.py --database mydb --input-dir ./backups --yes
        """
    )
    
    parser.add_argument('--database', required=True, help='Database name to restore to')
    parser.add_argument('--input-dir', required=True, help='Input directory with backup files')
    parser.add_argument('--containers', nargs='+', help='Specific containers to restore (default: all)')
    parser.add_argument('--overwrite', action='store_true', 
                       help='Overwrite existing items (default: skip existing)')
    parser.add_argument('--yes', '-y', action='store_true', help='Skip confirmation prompt')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Get Cosmos DB ID from environment
    cosmos_db_id = os.environ.get('AZURE_DB_ID')
    if not cosmos_db_id:
        logger.error("AZURE_DB_ID environment variable is not set")
        sys.exit(1)
    
    # Verify input directory exists
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    # Initialize restore
    restore = CosmosDBRestore(cosmos_db_id)
    
    if not restore.connect():
        logger.error("Failed to connect to Cosmos DB")
        sys.exit(1)
    
    # Get number of containers to restore
    if args.containers:
        num_containers = len(args.containers)
    else:
        backup_files = list(input_dir.glob("*.json"))
        backup_files = [f for f in backup_files if not f.name.startswith('_')]
        num_containers = len(backup_files)
    
    # Confirm restore
    if not args.yes:
        if not confirm_restore(args.database, num_containers, args.overwrite):
            logger.info("Restore cancelled by user")
            sys.exit(0)
    
    # Perform restore
    logger.info("="*70)
    logger.info(f"Starting restore to database '{args.database}'")
    logger.info("="*70)
    
    stats = restore.restore_database(
        args.database,
        input_dir,
        args.containers,
        args.overwrite
    )
    
    # Print summary
    print("\n" + "="*70)
    print("RESTORE SUMMARY")
    print("="*70)
    print(f"Database:        {stats['database']}")
    print(f"Containers:      {len(stats['containers'])}")
    print(f"Items restored:  {stats['total_restored']}")
    print(f"Items skipped:   {stats['total_skipped']}")
    print(f"Failed items:    {stats['total_failures']}")
    print("="*70)
    
    # Print container details
    print("\nContainer Details:")
    for container_name, container_stats in stats['containers'].items():
        status = "✓" if container_stats['failed_items'] == 0 else "⚠"
        print(f"  {status} {container_name}: {container_stats['items_restored']} restored, "
              f"{container_stats['items_skipped']} skipped, {container_stats['failed_items']} failed")
        if container_stats['errors']:
            for error in container_stats['errors'][:3]:
                print(f"      Error: {error}")
    
    print("\n" + "="*70)
    
    if stats['total_failures'] > 0:
        logger.warning("Restore completed with some failures")
        sys.exit(1)
    else:
        logger.info("Restore completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()

