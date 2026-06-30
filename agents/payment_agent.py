"""
Payment Agent - handles financial transactions
"""
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import random
import stripe

logger = logging.getLogger(__name__)


class PaymentAgent:
    """Agent for payment operations"""
    
    def __init__(self, stripe_api_key=None):
        self.transactions = []
        self.wallet_balance = 10000.00  # Test balance
        self.stripe_api_key = stripe_api_key
        self.logger = logger
        
        if stripe_api_key:
            stripe.api_key = stripe_api_key
    
    async def send_money(self, recipient: str, amount: float, description: str = '') -> Dict[str, Any]:
        """Send money to recipient"""
        try:
            if amount <= 0:
                return {'success': False, 'error': 'Amount must be greater than 0'}
            
            # Use Stripe if API key is available
            if self.stripe_api_key:
                try:
                    # Create a transfer or payment intent
                    # This is a simplified implementation - you'd need to set up proper Stripe flows
                    payment_intent = stripe.PaymentIntent.create(
                        amount=int(amount * 100),  # Convert to cents
                        currency='usd',
                        description=description or f'Payment to {recipient}',
                        metadata={'recipient': recipient}
                    )
                    
                    transaction = {
                        'id': payment_intent.id,
                        'timestamp': datetime.now().isoformat(),
                        'type': 'send',
                        'recipient': recipient,
                        'amount': amount,
                        'description': description,
                        'status': 'pending_payment',
                        'stripe_payment_intent_id': payment_intent.id
                    }
                    self.transactions.append(transaction)
                    
                    self.logger.info(f"Stripe payment created for {recipient}: {amount}")
                    return {
                        'success': True,
                        'message': f'Payment intent created for {amount} to {recipient}',
                        'transaction_id': transaction['id'],
                        'payment_intent_id': payment_intent.id,
                        'transaction': transaction
                    }
                except Exception as stripe_error:
                    self.logger.error(f"Stripe error: {str(stripe_error)}")
                    # Fall back to mock implementation
            else:
                # Fallback to mock implementation
                if amount > self.wallet_balance:
                    return {'success': False, 'error': f'Insufficient balance. Available: {self.wallet_balance}'}
                
                transaction = {
                    'id': f'TXN_{random.randint(100000, 999999)}',
                    'timestamp': datetime.now().isoformat(),
                    'type': 'send',
                    'recipient': recipient,
                    'amount': amount,
                    'description': description,
                    'status': 'completed'
                }
                self.transactions.append(transaction)
                self.wallet_balance -= amount
                
                self.logger.info(f"Payment sent (mock) to {recipient}: {amount}")
                return {
                    'success': True,
                    'message': f'Successfully sent {amount} to {recipient}',
                    'transaction_id': transaction['id'],
                    'new_balance': self.wallet_balance,
                    'transaction': transaction
                }
        except Exception as e:
            self.logger.error(f"Error sending money: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def request_money(self, requester: str, amount: float, reason: str = '') -> Dict[str, Any]:
        """Request money from someone"""
        try:
            request = {
                'id': f'REQ_{random.randint(100000, 999999)}',
                'timestamp': datetime.now().isoformat(),
                'type': 'request',
                'requester': requester,
                'amount': amount,
                'reason': reason,
                'status': 'pending'
            }
            self.transactions.append(request)
            
            self.logger.info(f"Payment request from {requester}: {amount}")
            return {
                'success': True,
                'message': f'Payment request sent to {requester} for {amount}',
                'request_id': request['id'],
                'request': request
            }
        except Exception as e:
            self.logger.error(f"Error requesting money: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def check_balance(self) -> Dict[str, Any]:
        """Check wallet balance"""
        try:
            return {
                'success': True,
                'balance': self.wallet_balance,
                'currency': 'KES',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error checking balance: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def get_transaction_history(self, limit: int = 10) -> Dict[str, Any]:
        """Get transaction history"""
        try:
            recent_transactions = self.transactions[-limit:] if self.transactions else []
            return {
                'success': True,
                'transaction_count': len(recent_transactions),
                'transactions': recent_transactions
            }
        except Exception as e:
            self.logger.error(f"Error getting transaction history: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        """Execute payment action"""
        if action == 'send':
            return await self.send_money(**kwargs)
        elif action == 'request':
            return await self.request_money(**kwargs)
        elif action == 'balance':
            return await self.check_balance(**kwargs)
        elif action == 'history':
            return await self.get_transaction_history(**kwargs)
        else:
            return {'success': False, 'error': f'Unknown action: {action}'}
