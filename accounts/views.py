from django.shortcuts import render, redirect
from .models import CustomUser
from .forms import UserForm  
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from home.models import PDF

def createUser(request):
    # Prevent logged-in users from accessing signup
    if request.user.is_authenticated:
        return redirect('home:home')

    if request.method == "POST":
        form = UserForm(request.POST)
        if form.is_valid():
            user = form.save()
            raw_password = form.cleaned_data.get('password1')  # Or 'password' if single field
            login(request, user)  # You can log in directly without authenticate()
            return redirect('home:home')
    else:
        form = UserForm()

    return render(request, 'accounts/createUser.html', {'form': form})


def loginUser(request):
    # Prevent logged-in users from seeing login page
    if request.user.is_authenticated:
        return redirect('home:home')

    if request.method == "POST":
        email = request.POST.get('email')
        password = request.POST.get('password')

        user = authenticate(request, email=email, password=password)

        if user is not None:
            login(request, user)
            return redirect('home:home')  # Redirect instead of render
        else:
            messages.error(request, "Invalid email or password.")

    return render(request, 'accounts/login.html')



@login_required(login_url='login')
def logoutUser(request):
    logout(request)  # This clears the session
    return redirect('home:home')


@login_required(login_url='login')
def updateUser(request):
    pdfs = PDF.objects.filter(user=request.user)
    if request.method == "POST":
        form = UserForm(request.POST,request.FILES, instance=request.user)
        if form.is_valid():
            form.save()
            messages.success(request, "🎉 Profile updated successfully!")
            return redirect('home')   # or 'home:home' if namespaced
        else:
            messages.error(request, "⚠️ Please correct the errors below.")
    else:
        form = UserForm(instance=request.user)

    return render(request, 'accounts/user.html', {'form': form, 'pdfs': pdfs})