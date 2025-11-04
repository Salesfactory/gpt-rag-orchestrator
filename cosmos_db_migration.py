#!/usr/bin/env python3
"""
Cosmos DB Container Migration Script

This script migrates all items from a source container in one database
to a destination container in another database within the same Azure Cosmos DB account.

Usage:
    python cosmos_db_migration.py --source-db <source_db_name> --dest-db <dest_db_name> --container <container_name>
    
    Or with different container names (requires confirmation):
    python cosmos_db_migration.py --source-db <source_db_name> --dest-db <dest_db_name> --source-container <source_name> --dest-container <dest_name> --allow-different-names

Environment Variables Required:
    - AZURE_DB_ID: The Cosmos DB account ID
    - AZURE_DB_NAME: The default database name (optional if --source-db is provided)
"""

import os
import sys
import argparse
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from azure.cosmos import CosmosClient, exceptions
from azure.identity import DefaultAzureCredential

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'migration_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


class CosmosDBMigrator:
    """Handles migration of items between Cosmos DB containers."""
    
    def __init__(self, cosmos_db_id: str):
        """
        Initialize the migrator.
        
        Args:
            cosmos_db_id: The Cosmos DB account ID
        """
        self.cosmos_db_uri = f"https://{cosmos_db_id}.documents.azure.com:443/"
        self.credential = DefaultAzureCredential()
        self.client = None
        
    def connect(self) -> bool:
        """
        Establish connection to Cosmos DB.
        
        Returns:
            True if connection successful, False otherwise
        """
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
    
    def verify_container_exists(self, database_name: str, container_name: str) -> bool:
        """
        Verify that a container exists in the specified database.
        
        Args:
            database_name: Name of the database
            container_name: Name of the container
            
        Returns:
            True if container exists, False otherwise
        """
        try:
            db = self.client.get_database_client(database_name)
            container = db.get_container_client(container_name)
            # Try to read container properties to verify it exists
            container.read()
            logger.info(f"Verified container '{container_name}' exists in database '{database_name}'")
            return True
        except exceptions.CosmosResourceNotFoundError:
            logger.error(f"Container '{container_name}' not found in database '{database_name}'")
            return False
        except Exception as e:
            logger.error(f"Error verifying container: {e}")
            return False
    
    def get_item_count(self, database_name: str, container_name: str) -> Optional[int]:
        """
        Get the approximate count of items in a container.
        
        Args:
            database_name: Name of the database
            container_name: Name of the container
            
        Returns:
            Number of items or None if error
        """
        try:
            db = self.client.get_database_client(database_name)
            container = db.get_container_client(container_name)
            
            query = "SELECT VALUE COUNT(1) FROM c"
            items = list(container.query_items(query=query, enable_cross_partition_query=True))
            count = items[0] if items else 0
            
            logger.info(f"Container '{container_name}' in database '{database_name}' contains {count} items")
            return count
        except Exception as e:
            logger.error(f"Error getting item count: {e}")
            return None
    
    def migrate_items(
        self,
        source_db: str,
        source_container: str,
        dest_db: str,
        dest_container: str,
        batch_size: int = 100,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Migrate all items from source to destination container.
        
        Args:
            source_db: Source database name
            source_container: Source container name
            dest_db: Destination database name
            dest_container: Destination container name
            batch_size: Number of items to process before logging progress
            dry_run: If True, only simulate the migration without writing
            
        Returns:
            Dictionary with migration statistics
        """
        stats = {
            'total_items': 0,
            'migrated_items': 0,
            'failed_items': 0,
            'skipped_items': 0,
            'errors': []
        }
        
        try:
            # Get database and container clients
            source_db_client = self.client.get_database_client(source_db)
            source_container_client = source_db_client.get_container_client(source_container)
            
            dest_db_client = self.client.get_database_client(dest_db)
            dest_container_client = dest_db_client.get_container_client(dest_container)
            
            logger.info(f"Starting migration from {source_db}/{source_container} to {dest_db}/{dest_container}")
            
            if dry_run:
                logger.warning("DRY RUN MODE - No data will be written to destination")
            
            # Query all items from source container
            query = "SELECT * FROM c"
            items = source_container_client.query_items(
                query=query,
                enable_cross_partition_query=True
            )
            
            # Process items
            for item in items:
                stats['total_items'] += 1
                item_id = item.get('id', 'unknown')
                
                try:
                    if not dry_run:
                        # Upsert item to destination (will create or update)
                        dest_container_client.upsert_item(item)
                    
                    stats['migrated_items'] += 1
                    
                    # Log progress
                    if stats['migrated_items'] % batch_size == 0:
                        logger.info(f"Progress: {stats['migrated_items']} items migrated...")
                        
                except exceptions.CosmosHttpResponseError as e:
                    stats['failed_items'] += 1
                    error_msg = f"Failed to migrate item {item_id}: {str(e)}"
                    logger.error(error_msg)
                    stats['errors'].append(error_msg)
                    
                except Exception as e:
                    stats['failed_items'] += 1
                    error_msg = f"Unexpected error migrating item {item_id}: {str(e)}"
                    logger.error(error_msg)
                    stats['errors'].append(error_msg)
            
            logger.info("Migration completed!")
            logger.info(f"Total items found: {stats['total_items']}")
            logger.info(f"Successfully migrated: {stats['migrated_items']}")
            logger.info(f"Failed migrations: {stats['failed_items']}")
            
            if dry_run:
                logger.info("DRY RUN - No actual data was written")
            
            return stats
            
        except Exception as e:
            logger.error(f"Critical error during migration: {e}")
            stats['errors'].append(f"Critical error: {str(e)}")
            return stats


def confirm_migration(
    source_db: str,
    source_container: str,
    dest_db: str,
    dest_container: str,
    source_count: int,
    dest_count: int
) -> bool:
    """
    Ask user to confirm the migration operation.
    
    Args:
        source_db: Source database name
        source_container: Source container name
        dest_db: Destination database name
        dest_container: Destination container name
        source_count: Number of items in source
        dest_count: Number of items in destination
        
    Returns:
        True if user confirms, False otherwise
    """
    print("\n" + "="*70)
    print("MIGRATION CONFIRMATION")
    print("="*70)
    print(f"Source:      {source_db}/{source_container} ({source_count} items)")
    print(f"Destination: {dest_db}/{dest_container} ({dest_count} items)")
    print("\nWARNING: This operation will upsert all items from source to destination.")
    print("Existing items with the same ID will be overwritten!")
    print("="*70)
    
    response = input("\nDo you want to proceed? (yes/no): ").strip().lower()
    return response in ['yes', 'y']


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Migrate items between Cosmos DB containers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Migrate between databases with same container name
  python cosmos_db_migration.py --source-db db1 --dest-db db2 --container mycontainer
  
  # Migrate with different container names
  python cosmos_db_migration.py --source-db db1 --dest-db db2 --source-container old_name --dest-container new_name --allow-different-names
  
  # Dry run to test without writing
  python cosmos_db_migration.py --source-db db1 --dest-db db2 --container mycontainer --dry-run
  
  # Skip confirmation prompt
  python cosmos_db_migration.py --source-db db1 --dest-db db2 --container mycontainer --yes
        """
    )
    
    parser.add_argument('--source-db', required=True, help='Source database name')
    parser.add_argument('--dest-db', required=True, help='Destination database name')
    parser.add_argument('--container', help='Container name (if same in both databases)')
    parser.add_argument('--source-container', help='Source container name (if different from destination)')
    parser.add_argument('--dest-container', help='Destination container name (if different from source)')
    parser.add_argument('--allow-different-names', action='store_true', 
                       help='Allow migration between containers with different names')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Number of items to process before logging progress (default: 100)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Simulate migration without writing to destination')
    parser.add_argument('--yes', '-y', action='store_true',
                       help='Skip confirmation prompt')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Determine container names
    if args.container:
        source_container = args.container
        dest_container = args.container
    elif args.source_container and args.dest_container:
        source_container = args.source_container
        dest_container = args.dest_container
        
        if source_container != dest_container and not args.allow_different_names:
            logger.error("Container names are different. Use --allow-different-names to proceed.")
            sys.exit(1)
    else:
        logger.error("Either --container or both --source-container and --dest-container must be provided")
        parser.print_help()
        sys.exit(1)
    
    # Verify container names match if required
    if not args.allow_different_names and source_container != dest_container:
        logger.error(f"Container names must match: '{source_container}' != '{dest_container}'")
        logger.error("Use --allow-different-names flag to migrate between different container names")
        sys.exit(1)
    
    # Get Cosmos DB ID from environment
    cosmos_db_id = os.environ.get('AZURE_DB_ID')
    if not cosmos_db_id:
        logger.error("AZURE_DB_ID environment variable is not set")
        sys.exit(1)
    
    # Initialize migrator
    migrator = CosmosDBMigrator(cosmos_db_id)
    
    # Connect to Cosmos DB
    if not migrator.connect():
        logger.error("Failed to connect to Cosmos DB. Exiting.")
        sys.exit(1)
    
    # Verify source container exists
    if not migrator.verify_container_exists(args.source_db, source_container):
        logger.error(f"Source container does not exist: {args.source_db}/{source_container}")
        sys.exit(1)
    
    # Verify destination container exists
    if not migrator.verify_container_exists(args.dest_db, dest_container):
        logger.error(f"Destination container does not exist: {args.dest_db}/{dest_container}")
        logger.error("Please create the destination container before running migration")
        sys.exit(1)
    
    # Get item counts
    source_count = migrator.get_item_count(args.source_db, source_container)
    dest_count = migrator.get_item_count(args.dest_db, dest_container)
    
    if source_count is None or dest_count is None:
        logger.error("Failed to get item counts. Exiting.")
        sys.exit(1)
    
    if source_count == 0:
        logger.warning("Source container is empty. Nothing to migrate.")
        sys.exit(0)
    
    # Confirm migration
    if not args.yes and not args.dry_run:
        if not confirm_migration(
            args.source_db,
            source_container,
            args.dest_db,
            dest_container,
            source_count,
            dest_count
        ):
            logger.info("Migration cancelled by user")
            sys.exit(0)
    
    # Perform migration
    stats = migrator.migrate_items(
        args.source_db,
        source_container,
        args.dest_db,
        dest_container,
        batch_size=args.batch_size,
        dry_run=args.dry_run
    )
    
    # Print final statistics
    print("\n" + "="*70)
    print("MIGRATION SUMMARY")
    print("="*70)
    print(f"Total items processed: {stats['total_items']}")
    print(f"Successfully migrated: {stats['migrated_items']}")
    print(f"Failed migrations:     {stats['failed_items']}")
    print(f"Skipped items:         {stats['skipped_items']}")
    
    if stats['errors']:
        print(f"\nErrors encountered: {len(stats['errors'])}")
        print("Check the log file for details")
    
    print("="*70)
    
    # Exit with appropriate code
    if stats['failed_items'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()

