import json
import os
from django.shortcuts import render, redirect
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_POST
from .models import SearchHistory, SavedItem, PDF
from .utils import process_pdf, hybrid_search
import numpy as np

def home(request):
    query = request.GET.get("q", "").strip()
    results = []
    saved = []
    if query:
        if request.user.is_authenticated:
            SearchHistory.objects.create(user=request.user, query=query)

        docs = list(PDF.objects.all().values("id", "author", "title", "abstract", "summary", "keywords", "embedding" , "file"))
        results = hybrid_search(query, docs, user_id=request.user.id if request.user.is_authenticated else None)
        if len(results) > 0:
            saved = SavedItem.objects.filter(user=request.user).values_list("title", flat=True)
    return render(request, "home/home.html", {
        "query": query,
        "results": results,
        "saved": saved,
    })


@login_required(login_url="login")
@require_POST
def save_item(request, docid):
    """Save a search result item"""
    title = request.POST.get("title")
    link = request.POST.get("link")
    SavedItem.objects.create(user=request.user, title=title, link=link)
    #return back to http_referer
    return redirect(request.META.get("HTTP_REFERER", "home:home"))



@login_required(login_url="login")
def history_view(request):
    history = SearchHistory.objects.filter(user=request.user).order_by("-timestamp")[:200]
    return render(request, "home/history.html", {"history": history})


@login_required(login_url="login")
def saved_view(request):
    saved = SavedItem.objects.filter(user=request.user).order_by("-saved_at")
    return render(request, "home/saved.html", {"saved": saved})



@login_required(login_url="login")
def upload_doc(request):
    if request.method == "POST":
        uploaded_file = request.FILES.get("document")
        if not uploaded_file:
            return render(request, "home/upload_doc.html", {"error": "No file uploaded."})

        # Temporary path
        temp_dir = os.path.join(settings.MEDIA_ROOT, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, uploaded_file.name)

        try:
            # Save temporarily
            with open(temp_path, "wb+") as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)

            # Process PDF
            data = process_pdf(temp_path)
            if not data:
                raise ValueError("Failed to process document.")

            # Save in DB
            pdf = PDF.objects.create(
                user=request.user,
                author=data.get("author"),
                title=data.get("title"),
                abstract=data.get("abstract"),
                summary=data.get("summary"),
                keywords=data.get("keywords"),
                text=data.get("text"),
                embedding=data.get("embedding") or [],
                file=uploaded_file,
            )

            return redirect("home:home")

        except Exception as e:
            return render(request, "home/upload_doc.html", {"error": str(e)})

        finally:
            # ✅ Always clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # GET request → show upload page
    return render(request, "home/upload_doc.html")