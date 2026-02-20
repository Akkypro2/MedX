package com.example.geminichatbot

import com.google.gson.annotations.SerializedName

data class MedicalQueryRequest(
    val query: String,
    @SerializedName("conversation_history")
    val conversation_history: List<HistoryItem>,
    @SerializedName("session_id")
    val sessionId: String
)

data class HistoryItem(
    @SerializedName("role")
    val role: String,    // Must be "user" or "model"

    @SerializedName("content")
    val content: String  // The actual message text
)
