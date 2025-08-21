from django.urls import path
from . import views

urlpatterns = [
    path('create-account/', views.createUser, name="create-account"),
    path('login/', views.loginUser, name="login"), 
    path('logout/' , view=views.logoutUser , name='logout'),
    path('user/' , views.updateUser , name='user'),  
]