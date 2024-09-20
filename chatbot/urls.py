from django.urls import path
from .views import ScrapeAPIView, QueryAPIView, TelegramWebhookAPIView

urlpatterns = [
    path('api/scrape/', ScrapeAPIView.as_view(), name='scrape'),
    path('api/query/', QueryAPIView.as_view(), name='query'),
    path('telegram/webhook/', TelegramWebhookAPIView.as_view(), name='telegram_webhook'),
]
