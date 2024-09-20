# chatbot/views.py
import logging
import requests

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings

from .helpers import process_query
from .utils import FAQScraper, prepare_data, get_embeddings, store_embeddings, search_embeddings
from .models import Team
from openai import OpenAI

client = OpenAI(api_key=settings.OPENAI_API_KEY)


logger = logging.getLogger(__name__)


class ScrapeAPIView(APIView):
    def post(self, request):
        team_id = request.data.get('team_id')
        url = request.data.get('url')
        custom_instructions = request.data.get('custom_instructions', '')

        if not team_id or not url:
            logger.error("team_id and url are required.")
            return Response({'error': 'team_id and url are required.'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Get or create the team
            team, created = Team.objects.get_or_create(team_id=team_id)
            if custom_instructions:
                team.custom_instructions = custom_instructions
                team.save()
                logger.debug(f"Updated custom instructions for team {team_id}")

            # Scrape and process data
            scraper = FAQScraper(url, max_pages=20)  # Adjust max_pages as needed
            scraper.scrape()

            if not scraper.data:
                logger.error("No FAQs were scraped from the URL.")
                return Response({'error': 'No FAQs were scraped from the URL.'}, status=status.HTTP_400_BAD_REQUEST)

            prepared_data = prepare_data(scraper.data)

            # Generate embeddings
            texts = [item['text'] for item in prepared_data]
            if not texts:
                logger.error("No texts to generate embeddings.")
                return Response({'error': 'No text data was extracted from the FAQs.'},
                                status=status.HTTP_400_BAD_REQUEST)

            embeddings = get_embeddings(texts)

            # Store embeddings in Qdrant
            store_embeddings(team_id, embeddings, texts)

            logger.info(f"Data for team {team_id} has been updated.")
            return Response({'message': f'Data for team {team_id} has been updated.'}, status=status.HTTP_200_OK)

        except Exception as e:
            logger.exception("An error occurred during the scraping process.")
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class QueryAPIView(APIView):
    def post(self, request):
        team_id = request.data.get('team_id')
        message = request.data.get('message')

        if not team_id or not message:
            logger.error("team_id and message are required.")
            return Response({'error': 'team_id and message are required.'}, status=status.HTTP_400_BAD_REQUEST)

        # Hardcode team_id to 123 if not provided (as per your request)
        if not team_id:
            team_id = 123

        try:
            # Check if the team exists (optional if you have a Team model)
            # If you want to skip this, you can remove this block
            try:
                team = Team.objects.get(team_id=team_id)
                custom_instructions = team.custom_instructions
            except Team.DoesNotExist:
                logger.error(f"Invalid team_id: {team_id}")
                custom_instructions = None  # Or set to default instructions

            answer = process_query(team_id, message, custom_instructions)

            return Response({'answer': answer}, status=status.HTTP_200_OK)

        except Exception as e:
            logger.exception("An error occurred during the query process.")
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class TelegramWebhookAPIView(APIView):

    def post(self, request):
        try:
            update = request.data
            logger.debug(f"Received update: {update}")

            # Extract chat_id and message
            message = update.get('message')
            if not message:
                logger.error("No message found in update.")
                return Response(status=status.HTTP_200_OK)  # Telegram requires a 200 response even if no message

            chat = message.get('chat')
            if not chat:
                logger.error("No chat information found in message.")
                return Response(status=status.HTTP_200_OK)

            chat_id = chat.get('id')
            message_text = message.get('text')

            if not message_text:
                logger.error("No text found in message.")
                return Response(status=status.HTTP_200_OK)

            # Hardcode team_id to 123
            team_id = 123

            # Optional: Set custom instructions or leave as None
            custom_instructions = None  # Or provide a default string

            # Process the query directly using the helper function
            answer = process_query(team_id, message_text, custom_instructions)

            # Send the answer back to the user via Telegram
            self.send_telegram_message(chat_id, answer)
            return Response(status=status.HTTP_200_OK)

        except Exception as e:
            logger.exception("An error occurred in TelegramWebhookAPIView.")
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def send_telegram_message(self, chat_id, text):
        token = settings.TELEGRAM_BOT_API_KEY
        url = f"https://api.telegram.org/{token}/sendMessage"
        logger.debug(f"chat id: {chat_id}, answer: {text}, token: {token}")
        payload = {
            'chat_id': str(chat_id),
            'text': text,
            'parse_mode': 'Markdown'
        }
        response = requests.post(url, data=payload)
        if response.status_code != 200:
            logger.error(f"Failed to send message: {response.text}")
