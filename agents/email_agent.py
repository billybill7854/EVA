"""
Email Agent - handles email sending and reading
"""
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials

logger = logging.getLogger(__name__)


class EmailAgent:
    """Agent for email operations"""
    
    def __init__(self, client_id=None, client_secret=None):
        self.emails_sent = []
        self.emails_received = []
        self.client_id = client_id
        self.client_secret = client_secret
        self.gmail_service = None
        self.logger = logger
    
    async def send_email(self, to: str, subject: str, body: str, attachments: Optional[list] = None) -> Dict[str, Any]:
        """Send an email"""
        try:
            # Note: Real Gmail API integration requires OAuth 2.0 flow
            # This is a placeholder implementation
            # For production, you need to:
            # 1. Set up OAuth 2.0 consent screen in Google Cloud Console
            # 2. Obtain user authorization and refresh tokens
            # 3. Use the tokens to create Credentials object
            # 4. Build Gmail service and use it to send emails
            
            email_record = {
                'timestamp': datetime.now().isoformat(),
                'to': to,
                'subject': subject,
                'body': body,
                'attachments': attachments or [],
                'status': 'sent'
            }
            self.emails_sent.append(email_record)
            self.logger.info(f"Email sent to {to}: {subject}")
            return {
                'success': True,
                'message': f'Email sent to {to}',
                'email_id': len(self.emails_sent),
                'timestamp': email_record['timestamp']
            }
        except Exception as e:
            self.logger.error(f"Error sending email: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def read_emails(self, inbox: str = 'INBOX', limit: int = 10) -> Dict[str, Any]:
        """Read emails from inbox"""
        try:
            # Simulate reading from inbox
            recent_emails = self.emails_received[-limit:] if self.emails_received else []
            return {
                'success': True,
                'inbox': inbox,
                'email_count': len(recent_emails),
                'emails': recent_emails
            }
        except Exception as e:
            self.logger.error(f"Error reading emails: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def search_emails(self, query: str, sender: Optional[str] = None) -> Dict[str, Any]:
        """Search emails"""
        try:
            results = []
            all_emails = self.emails_sent + self.emails_received
            
            for email in all_emails:
                if query.lower() in email.get('subject', '').lower() or \
                   query.lower() in email.get('body', '').lower():
                    if sender is None or sender in email.get('to', '') or sender in email.get('from', ''):
                        results.append(email)
            
            return {
                'success': True,
                'query': query,
                'results_count': len(results),
                'emails': results
            }
        except Exception as e:
            self.logger.error(f"Error searching emails: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        """Execute email action"""
        if action == 'send':
            return await self.send_email(**kwargs)
        elif action == 'read':
            return await self.read_emails(**kwargs)
        elif action == 'search':
            return await self.search_emails(**kwargs)
        else:
            return {'success': False, 'error': f'Unknown action: {action}'}
