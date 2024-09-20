import logging

from openai import OpenAI

from django.conf import settings

from chatbot.utils import get_embeddings, search_embeddings

logger = logging.getLogger(__name__)

client = OpenAI(api_key=settings.OPENAI_API_KEY)


def process_query(team_id, message, custom_instructions=None):
    try:
        # Generate embedding for the query
        query_embedding = get_embeddings([message])[0]

        # Search for relevant texts
        relevant_texts = search_embeddings(team_id, query_embedding, top_k=5)

        if not relevant_texts:
            logger.warning(f"No relevant texts found for team {team_id}")
            return "I'm sorry, I couldn't find relevant information for your query."

        # Build context
        context = "\n\n".join(relevant_texts)

        # Add custom instructions
        system_prompt = custom_instructions or "You are a helpful assistant."

        # Generate response
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{message}"},
            ],
            max_tokens=256,
            temperature=0.7
        )
        answer = response.choices[0].message.content.strip()

        logger.info(f"Generated response for team {team_id}")
        return answer

    except Exception as e:
        logger.exception("An error occurred during the query process.")
        return "An error occurred while processing your request."
