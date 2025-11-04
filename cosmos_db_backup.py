#!/usr/bin/env python3
"""
Cosmos DB Database Backup Script

Backs up all containers from a Cosmos DB database to JSON files.

Usage:
    python cosmos_db_backup.py --database <db_name> --output-dir <backup_dir>
    python cosmos_db_backup.py --database <db_name> --output-dir <backup_dir> --containers container1 container2
"""

import os
import sys
import json
import argparse
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from azure.cosmos import CosmosClient, exceptions
from azure.identity import DefaultAzureCredential

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class CosmosDBBackup:
    """Handles backup of Cosmos DB containers to JSON files."""
    
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
    
    def list_containers(self, database_name: str) -> List[str]:
        """List all containers in a database."""
        try:
            db = self.client.get_database_client(database_name)
            containers = list(db.list_containers())
            container_names = [c['id'] for c in containers]
            logger.info(f"Found {len(container_names)} containers in database '{database_name}'")
            return container_names
        except Exception as e:
            logger.error(f"Error listing containers: {e}")
            return []
    
    def backup_container(
        self,
        database_name: str,
        container_name: str,
        output_file: Path
    ) -> Dict[str, Any]:
        """Backup a single container to a JSON file."""
        stats = {
            'container': container_name,
            'items_backed_up': 0,
            'failed_items': 0,
            'file_path': str(output_file),
            'errors': []
        }
        
        try:
            db = self.client.get_database_client(database_name)
            container = db.get_container_client(container_name)
            
            logger.info(f"Backing up container '{container_name}'...")
            
            # Query all items
            query = "SELECT * FROM c"
            items = container.query_items(
                query=query,
                enable_cross_partition_query=True
            )
            
            # Collect all items
            all_items = []
            for item in items:
                try:
                    all_items.append(item)
                    stats['items_backed_up'] += 1
                    
                    if stats['items_backed_up'] % 100 == 0:
                        logger.info(f"  Progress: {stats['items_backed_up']} items...")
                        
                except Exception as e:
                    stats['failed_items'] += 1
                    error_msg = f"Failed to read item: {str(e)}"
                    stats['errors'].append(error_msg)
                    logger.error(f"  {error_msg}")
            
            # Write to file
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_items, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✓ Backed up {stats['items_backed_up']} items from '{container_name}' to {output_file}")
            
        except Exception as e:
            error_msg = f"Critical error backing up container '{container_name}': {str(e)}"
            logger.error(error_msg)
            stats['errors'].append(error_msg)
        
        return stats
    
    def backup_database(
        self,
        database_name: str,
        output_dir: Path,
        containers: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Backup entire database or specific containers."""
        backup_stats = {
            'database': database_name,
            'timestamp': datetime.now().isoformat(),
            'containers': {},
            'total_items': 0,
            'total_failures': 0
        }
        
        # Get list of containers to backup
        if containers:
            container_list = containers
            logger.info(f"Backing up {len(container_list)} specified containers")
        else:
            container_list = self.list_containers(database_name)
            logger.info(f"Backing up all {len(container_list)} containers from database")
        
        if not container_list:
            logger.error("No containers to backup")
            return backup_stats
        
        # Backup each container
        for container_name in container_list:
            output_file = output_dir / f"{container_name}.json"
            stats = self.backup_container(database_name, container_name, output_file)
            
            backup_stats['containers'][container_name] = stats
            backup_stats['total_items'] += stats['items_backed_up']
            backup_stats['total_failures'] += stats['failed_items']
        
        # Save metadata
        metadata_file = output_dir / '_backup_metadata.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(backup_stats, f, indent=2)
        
        logger.info(f"Backup metadata saved to {metadata_file}")
        
        return backup_stats


def main():
    parser = argparse.ArgumentParser(
        description='Backup Cosmos DB database to JSON files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Backup entire database
  python cosmos_db_backup.py --database mydb --output-dir ./backups/mydb_backup
  
  # Backup specific containers only
  python cosmos_db_backup.py --database mydb --output-dir ./backups --containers conversations schedules
  
  # Backup with custom timestamp
  python cosmos_db_backup.py --database mydb --output-dir ./backups/backup_$(date +%Y%m%d_%H%M%S)
        """
    )
    
    parser.add_argument('--database', required=True, help='Database name to backup')
    parser.add_argument('--output-dir', required=True, help='Output directory for backup files')
    parser.add_argument('--containers', nargs='+', help='Specific containers to backup (default: all)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Get Cosmos DB ID from environment
    cosmos_db_id = os.environ.get('AZURE_DB_ID')
    if not cosmos_db_id:
        logger.error("AZURE_DB_ID environment variable is not set")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Backup directory: {output_dir.absolute()}")
    
    # Initialize backup
    backup = CosmosDBBackup(cosmos_db_id)
    
    if not backup.connect():
        logger.error("Failed to connect to Cosmos DB")
        sys.exit(1)
    
    # Perform backup
    logger.info("="*70)
    logger.info(f"Starting backup of database '{args.database}'")
    logger.info("="*70)
    
    stats = backup.backup_database(
        args.database,
        output_dir,
        args.containers
    )
    
    # Print summary
    print("\n" + "="*70)
    print("BACKUP SUMMARY")
    print("="*70)
    print(f"Database:        {stats['database']}")
    print(f"Timestamp:       {stats['timestamp']}")
    print(f"Containers:      {len(stats['containers'])}")
    print(f"Total items:     {stats['total_items']}")
    print(f"Failed items:    {stats['total_failures']}")
    print(f"Output dir:      {output_dir.absolute()}")
    print("="*70)
    
    # Print container details
    print("\nContainer Details:")
    for container_name, container_stats in stats['containers'].items():
        status = "✓" if container_stats['failed_items'] == 0 else "⚠"
        print(f"  {status} {container_name}: {container_stats['items_backed_up']} items")
        if container_stats['errors']:
            for error in container_stats['errors'][:3]:  # Show first 3 errors
                print(f"      Error: {error}")
    
    print("\n" + "="*70)
    
    if stats['total_failures'] > 0:
        logger.warning("Backup completed with some failures")
        sys.exit(1)
    else:
        logger.info("Backup completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()

