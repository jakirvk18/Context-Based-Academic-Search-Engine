from django.db import models
from accounts.models import CustomUser as User
from django.contrib.postgres.fields import ArrayField


class SearchHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    query = models.CharField(max_length=255)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.email} — {self.query[:60]}"


class SavedItem(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    title = models.CharField(max_length=255)
    link = models.URLField()
    saved_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.title} ({self.user.email})"


class PDF(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    author = models.CharField(max_length=255, blank=True, null=True)
    title = models.CharField(max_length=500)
    abstract = models.TextField(blank=True, null=True)
    summary = models.TextField(blank=True, null=True)
    keywords = models.TextField(blank=True, null=True)
    text = models.TextField(blank=True, null=True)
    embedding = models.JSONField(default=list, blank=True)   # ✅ Works in SQLite & Postgres
    file = models.FileField(upload_to="documents/")

    def __str__(self):
        return self.title
