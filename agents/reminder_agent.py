"""
Reminder Agent - handles reminders and notifications
"""
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ReminderAgent:
    """Agent for reminder operations"""
    
    def __init__(self):
        self.reminders = []
        self.logger = logger
    
    async def set_reminder(self, title: str, date: str, time: str, 
                          description: str = '', priority: str = 'normal') -> Dict[str, Any]:
        """Set a reminder"""
        try:
            reminder = {
                'id': len(self.reminders) + 1,
                'title': title,
                'date': date,
                'time': time,
                'description': description,
                'priority': priority,
                'created_at': datetime.now().isoformat(),
                'status': 'active'
            }
            self.reminders.append(reminder)
            self.logger.info(f"Reminder set: {title} on {date} at {time}")
            return {
                'success': True,
                'message': f'Reminder set for {date} at {time}',
                'reminder_id': reminder['id'],
                'reminder': reminder
            }
        except Exception as e:
            self.logger.error(f"Error setting reminder: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def list_reminders(self, status: str = 'active') -> Dict[str, Any]:
        """List all reminders"""
        try:
            filtered = [r for r in self.reminders if r['status'] == status]
            return {
                'success': True,
                'reminder_count': len(filtered),
                'status_filter': status,
                'reminders': filtered
            }
        except Exception as e:
            self.logger.error(f"Error listing reminders: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def update_reminder(self, reminder_id: int, **updates) -> Dict[str, Any]:
        """Update a reminder"""
        try:
            for reminder in self.reminders:
                if reminder['id'] == reminder_id:
                    reminder.update(updates)
                    self.logger.info(f"Reminder {reminder_id} updated")
                    return {
                        'success': True,
                        'message': f'Reminder {reminder_id} updated',
                        'reminder': reminder
                    }
            return {'success': False, 'error': f'Reminder {reminder_id} not found'}
        except Exception as e:
            self.logger.error(f"Error updating reminder: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def complete_reminder(self, reminder_id: int) -> Dict[str, Any]:
        """Mark reminder as completed"""
        try:
            for reminder in self.reminders:
                if reminder['id'] == reminder_id:
                    reminder['status'] = 'completed'
                    reminder['completed_at'] = datetime.now().isoformat()
                    self.logger.info(f"Reminder {reminder_id} completed")
                    return {
                        'success': True,
                        'message': f'Reminder {reminder_id} marked as completed'
                    }
            return {'success': False, 'error': f'Reminder {reminder_id} not found'}
        except Exception as e:
            self.logger.error(f"Error completing reminder: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def delete_reminder(self, reminder_id: int) -> Dict[str, Any]:
        """Delete a reminder"""
        try:
            for i, reminder in enumerate(self.reminders):
                if reminder['id'] == reminder_id:
                    self.reminders.pop(i)
                    self.logger.info(f"Reminder {reminder_id} deleted")
                    return {
                        'success': True,
                        'message': f'Reminder {reminder_id} deleted'
                    }
            return {'success': False, 'error': f'Reminder {reminder_id} not found'}
        except Exception as e:
            self.logger.error(f"Error deleting reminder: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        """Execute reminder action"""
        if action == 'set':
            return await self.set_reminder(**kwargs)
        elif action == 'list':
            return await self.list_reminders(**kwargs)
        elif action == 'update':
            return await self.update_reminder(**kwargs)
        elif action == 'complete':
            return await self.complete_reminder(**kwargs)
        elif action == 'delete':
            return await self.delete_reminder(**kwargs)
        else:
            return {'success': False, 'error': f'Unknown action: {action}'}
