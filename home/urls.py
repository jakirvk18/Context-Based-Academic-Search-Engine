from django.urls import path
from . import views

app_name = 'home'

urlpatterns = [
    path('', views.home, name='home'),
    path('save/<str:docid>/', views.save_item, name='save_item'),
    path('history/', views.history_view, name='history'),
    path('saved/', views.saved_view, name='saved'),
    path('upload-doc/', views.upload_doc, name='upload_doc')
]