from django.db import models


class Team(models.Model):
    team_id = models.CharField(max_length=100, unique=True)
    custom_instructions = models.TextField(blank=True, null=True)

    def __str__(self):
        return self.team_id
