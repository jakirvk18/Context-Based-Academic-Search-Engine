from django.contrib import admin
from .models import SavedItem, SearchHistory , PDF

admin.site.register(SavedItem)
admin.site.register(SearchHistory)
admin.site.register(PDF)