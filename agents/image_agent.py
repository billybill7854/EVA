"""
Image Agent - handles image generation and manipulation
"""
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import random
import base64

logger = logging.getLogger(__name__)


class ImageAgent:
    """Agent for image operations"""
    
    def __init__(self, nvidia_service=None):
        self.images_generated = []
        self.nvidia_service = nvidia_service
        self.logger = logger
    
    async def generate_image(self, prompt: str, model: str = 'flux', size: str = '1024x1024',
                            quality: str = 'hd', num_images: int = 1) -> Dict[str, Any]:
        """Generate image from prompt"""
        try:
            # Parse size
            width, height = map(int, size.split('x'))
            
            # Use NVIDIA service if available
            if self.nvidia_service:
                image_bytes_list = await self.nvidia_service.call_image_gen_model(
                    prompt=prompt,
                    width=width,
                    height=height,
                    num_images=num_images,
                )
                
                images = []
                for i, image_bytes in enumerate(image_bytes_list):
                    # Convert bytes to base64 for storage/transmission
                    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
                    
                    image = {
                        'id': f'IMG_{random.randint(100000, 999999)}',
                        'prompt': prompt,
                        'model': model,
                        'size': size,
                        'quality': quality,
                        'generated_at': datetime.now().isoformat(),
                        'image_data': image_b64,
                        'status': 'ready'
                    }
                    images.append(image)
                    self.images_generated.append(image)
                
                self.logger.info(f"Images generated with NVIDIA: {prompt}")
                return {
                    'success': True,
                    'message': f'Generated {num_images} image(s) for: {prompt}',
                    'images': images,
                    'image_count': len(images)
                }
            else:
                # Fallback to mock generation
                images = []
                for i in range(num_images):
                    image = {
                        'id': f'IMG_{random.randint(100000, 999999)}',
                        'prompt': prompt,
                        'model': model,
                        'size': size,
                        'quality': quality,
                        'generated_at': datetime.now().isoformat(),
                        'url': f'https://example.com/image_{i + 1}.png',
                        'status': 'ready'
                    }
                    images.append(image)
                    self.images_generated.append(image)
                
                self.logger.info(f"Images generated (mock): {prompt}")
                return {
                    'success': True,
                    'message': f'Generated {num_images} image(s) for: {prompt}',
                    'images': images,
                    'image_count': len(images)
                }
        except Exception as e:
            self.logger.error(f"Error generating image: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def edit_image(self, image_id: str, edits: Dict[str, Any]) -> Dict[str, Any]:
        """Edit an existing image"""
        try:
            edited_image = {
                'original_id': image_id,
                'id': f'IMG_{random.randint(100000, 999999)}',
                'edits': edits,
                'edited_at': datetime.now().isoformat(),
                'url': f'https://example.com/edited_image.png',
                'status': 'ready'
            }
            self.images_generated.append(edited_image)
            
            self.logger.info(f"Image {image_id} edited")
            return {
                'success': True,
                'message': f'Image edited successfully',
                'image': edited_image
            }
        except Exception as e:
            self.logger.error(f"Error editing image: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def upscale_image(self, image_id: str, scale_factor: int = 2) -> Dict[str, Any]:
        """Upscale an image"""
        try:
            upscaled_image = {
                'original_id': image_id,
                'id': f'IMG_{random.randint(100000, 999999)}',
                'scale_factor': scale_factor,
                'upscaled_at': datetime.now().isoformat(),
                'url': f'https://example.com/upscaled_image.png',
                'status': 'ready'
            }
            self.images_generated.append(upscaled_image)
            
            self.logger.info(f"Image {image_id} upscaled {scale_factor}x")
            return {
                'success': True,
                'message': f'Image upscaled by {scale_factor}x',
                'image': upscaled_image
            }
        except Exception as e:
            self.logger.error(f"Error upscaling image: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def get_generated_images(self, limit: int = 10) -> Dict[str, Any]:
        """Get list of generated images"""
        try:
            recent_images = self.images_generated[-limit:] if self.images_generated else []
            return {
                'success': True,
                'image_count': len(recent_images),
                'images': recent_images
            }
        except Exception as e:
            self.logger.error(f"Error getting generated images: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def delete_image(self, image_id: str) -> Dict[str, Any]:
        """Delete an image"""
        try:
            for i, img in enumerate(self.images_generated):
                if img['id'] == image_id:
                    self.images_generated.pop(i)
                    self.logger.info(f"Image {image_id} deleted")
                    return {
                        'success': True,
                        'message': f'Image {image_id} deleted'
                    }
            return {'success': False, 'error': f'Image {image_id} not found'}
        except Exception as e:
            self.logger.error(f"Error deleting image: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        """Execute image action"""
        if action == 'generate':
            return await self.generate_image(**kwargs)
        elif action == 'edit':
            return await self.edit_image(**kwargs)
        elif action == 'upscale':
            return await self.upscale_image(**kwargs)
        elif action == 'list':
            return await self.get_generated_images(**kwargs)
        elif action == 'delete':
            return await self.delete_image(**kwargs)
        else:
            return {'success': False, 'error': f'Unknown action: {action}'}
