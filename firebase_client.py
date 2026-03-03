"""
Firebase client for state management and real-time data streaming.
Critical component for ecosystem persistence and distributed coordination.
"""
import os
import json
import logging
from typing import Any, Dict, Optional, List
from datetime import datetime

import firebase_admin
from firebase_admin import credentials, firestore, db
from firebase_admin.exceptions import FirebaseError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FirebaseClient:
    """
    Firebase client wrapper with comprehensive error handling and type safety.
    Ensures reliable state management across the trading ecosystem.
    """
    
    def __init__(self, project_id: str, service_account_path: str, database_url: str):
        """
        Initialize Firebase connection with robust error handling.
        
        Args:
            project_id: Firebase project ID
            service_account_path: Path to service account JSON
            database_url: Firebase database URL
        """
        self.project_id = project_id
        self.service_account_path = service_account_path
        self.database_url = database_url
        
        # Initialize Firebase app only if not already initialized
        if not firebase_admin._apps:
            try:
                if os.path.exists(service_account_path):
                    cred = credentials.Certificate(service_account_path)
                    self.app = firebase_admin.initialize_app(
                        cred,
                        {
                            'databaseURL': database_url,
                            'projectId': project_id
                        }
                    )
                    logger.info("Firebase app initialized successfully")
                else:
                    logger.error(f"Service account file not found: {service_account_path}")
                    raise FileNotFoundError(f"Service account file not found: {service_account_path}")
            except FirebaseError as e:
                logger.error(f"Firebase initialization failed: {str(e)}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error during Firebase initialization: {str(e)}