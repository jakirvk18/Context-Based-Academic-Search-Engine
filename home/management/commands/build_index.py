# search_engine/management/commands/build_index.py
from django.core.management.base import BaseCommand
from django.conf import settings
from home.indexing import build_index

class Command(BaseCommand):
    help = 'Build inverted index from documents/ folder'

    def handle(self, *args, **options):
        base = settings.BASE_DIR
        build_index(documents_dir='documents', base_dir=base)
        self.stdout.write(self.style.SUCCESS('Index built successfully.'))
