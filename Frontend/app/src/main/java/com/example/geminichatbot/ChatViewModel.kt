import android.app.Application
import android.content.Context
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.os.Build
import android.util.Log
import androidx.annotation.RequiresApi
import androidx.compose.runtime.mutableStateListOf
import com.example.geminichatbot.RetrofitClient
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.geminichatbot.Constants
import com.example.geminichatbot.HistoryItem
import com.example.geminichatbot.MedicalChatApi
import com.example.geminichatbot.MedicalQueryRequest
import com.example.geminichatbot.MedicalQueryResponse
import com.example.geminichatbot.MessageModel
import com.example.geminichatbot.ui.theme.MessageRow
import com.google.ai.client.generativeai.GenerativeModel
import com.google.ai.client.generativeai.type.BlockThreshold
import com.google.ai.client.generativeai.type.HarmCategory
import com.google.ai.client.generativeai.type.SafetySetting
import kotlinx.coroutines.launch
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import java.util.UUID
import kotlin.uuid.Uuid

class ChatViewModel(application: Application) : AndroidViewModel(application) {

    private val _messageList = mutableStateListOf<MessageModel>()
    val messageList: List<MessageModel> = _messageList

    private val _isTyping = MutableLiveData(false)
    val isTyping: LiveData<Boolean> = _isTyping

    private val apiService = RetrofitClient.instance
    private val sessionId: String = UUID.randomUUID().toString()


    private val safetySettings = listOf(
        SafetySetting(
            harmCategory = HarmCategory.HARASSMENT,
            threshold = BlockThreshold.MEDIUM_AND_ABOVE
        ),
        SafetySetting(
            harmCategory = HarmCategory.HATE_SPEECH,
            threshold = BlockThreshold.MEDIUM_AND_ABOVE
        ),
        SafetySetting(
            harmCategory = HarmCategory.SEXUALLY_EXPLICIT,
            threshold = BlockThreshold.LOW_AND_ABOVE
        ),
        SafetySetting(
            harmCategory = HarmCategory.DANGEROUS_CONTENT,
            threshold = BlockThreshold.MEDIUM_AND_ABOVE
        )
    )

    // Gemini model for content moderation
    private val moderationModel = GenerativeModel(
        modelName = "gemini-2.5-flash",
        apiKey = Constants.apiKey, // Use your existing API key
        safetySettings = safetySettings
    )

    // Blocked keywords list
    private val blockedKeywords = listOf(
        "offensive", "hate", "abusive", "violent", "fuck", "kill", "murder"
        // Add more keywords as needed
    )




    private fun isNetworkAvailable(): Boolean {
        val cm = getApplication<Application>().getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
        val network = cm.activeNetwork ?: return false
        val capabilities = cm.getNetworkCapabilities(network) ?: return false
        return capabilities.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET)
    }


    @RequiresApi(Build.VERSION_CODES.VANILLA_ICE_CREAM)
    fun sendMessage(question: String, context: Context) {
        viewModelScope.launch {
            try {
                // Add user message
                _messageList.add(MessageModel(message = question, role = "user"))

                // Check internet connection
                if (!isNetworkAvailable()) {
                    _messageList.add(MessageModel(
                        message = "No internet connection. Please check your network.",
                        role = "model"
                    ))
                    return@launch
                }

                if (containsBlockedKeywords(question)) {
                    _messageList.add(MessageModel(
                        message = "⚠️ Your message contains inappropriate language. Please rephrase your question respectfully.",
                        role = "model"
                    ))
                    return@launch
                }

                // Step 2: Check if medical-related (BEFORE any API calls)
                val isMedical = checkIfMedicalQuery(question)
                if (!isMedical) {
                    _messageList.add(MessageModel(
                        message = "ℹ️ This chatbot specializes in medical and health-related questions. Please ask about medical topics.",
                        role = "model"
                    ))
                    return@launch
                }

                // Step 3: Gemini safety check (BEFORE any API calls)
                val safetyCheckPassed = performSafetyCheck(question)
                if (!safetyCheckPassed) {
                    _messageList.add(MessageModel(
                        message = "⚠️ Your question couldn't be processed due to safety concerns. Please rephrase it in a respectful manner.",
                        role = "model"
                    ))
                    return@launch
                }

                // Add empty bot message that will be animated
                val botMessageIndex = messageList.size
                _messageList.add(MessageModel(message = "Thinking...", role = "model"))

                // Get full response from Gemini
                val response = withContext(Dispatchers.IO) {

                    // 1. Prepare the History
                    // We use dropLast(2) to remove:
                    //   - The last item: "Thinking..."
                    //   - The second to last item: The current "question" (because we send it separately in 'query')
                    val historyToSend = _messageList.dropLast(2).map { msg ->
                        HistoryItem(
                            role = msg.role,   // "user" or "model"
                            content = msg.message
                        )
                    }

                    // 2. Send the Request
                    apiService.sendMedicalQuery(
                        MedicalQueryRequest(
                            query = question,
                            conversation_history = historyToSend,
                            sessionId = this@ChatViewModel.sessionId
                        )
                    )
                }

                val formattedResponse = formatMedicalResponse(response)
                // Animate the response word by word
                animateTextResponse(formattedResponse, botMessageIndex)

            } catch (e: Exception) {
                if(_messageList.isNotEmpty()){
                    _messageList[_messageList.size - 1] = MessageModel(
                        "Something went wrong. Please try again",
                        "model"
                    )
                }else{
                    _messageList.add(MessageModel(
                        "Something went wrong. Please try again",
                        "model"
                    ))
                }
                e.printStackTrace()
            }
        }
    }

    private fun containsBlockedKeywords(text: String): Boolean {
        val lowerText = text.lowercase()
        return blockedKeywords.any { keyword ->
            lowerText.contains(keyword.lowercase())
        }
    }
    // Updating history feature

    private suspend fun checkIfMedicalQuery(query: String): Boolean {
        return try {
            withContext(Dispatchers.IO) {
                val validationPrompt = """
                    Is this a legitimate medical or health-related question?
                    Answer only with YES or NO.
                    
                    Question: $query
                """.trimIndent()

                val response = moderationModel.generateContent(validationPrompt)
                val answer = response.text?.uppercase() ?: "YES"
                answer.contains("YES")
            }
        } catch (e: Exception) {
            Log.e("ChatViewModel", "Medical validation error: ${e.message}")
            // If validation fails, allow the query to proceed
            true
        }
    }

    private suspend fun performSafetyCheck(query: String): Boolean {
        return try {
            withContext(Dispatchers.IO) {
                // Try to generate content with safety filters
                val response = moderationModel.generateContent(query)

                // Check if content was blocked
                if (response.candidates.isEmpty()) {
                    Log.w("ChatViewModel", "Content blocked by safety filters")
                    return@withContext false
                }

                val candidate = response.candidates.first()
                val isBlocked = candidate.finishReason.toString() == "SAFETY"

                if (isBlocked) {
                    val blockedCategories = response.promptFeedback?.safetyRatings
                        ?.map { it.category.name }
                    Log.w("ChatViewModel", "Blocked categories: $blockedCategories")
                    return@withContext false
                }

                true
            }
        } catch (e: Exception) {
            Log.e("ChatViewModel", "Safety check error: ${e.message}")
            // If safety check fails, allow the query (fail open for better UX)
            true
        }
    }


    private fun formatMedicalResponse(response: MedicalQueryResponse): String{
        var formattedText = response.response

        if(!response.sources.isNullOrEmpty()){
            formattedText += "\n\n sources\n"
            response.sources.forEach { source->
                formattedText += "- $source\n"
            }
        }
        return formattedText
    }
    private suspend fun animateTextResponse(fullText: String, messageIndex: Int) {
        val words = fullText.split(" ")
        var currentText = ""

        _isTyping.value = true

        for (word in words) {
            currentText += if (currentText.isEmpty()) word else " $word"

            // Update the message at the specific index
            _messageList[messageIndex] = _messageList[messageIndex].copy(message = currentText)

            // Delay between words (adjust speed here)
            delay(50) // 100ms between each word
        }

        _isTyping.value = false
    }

}