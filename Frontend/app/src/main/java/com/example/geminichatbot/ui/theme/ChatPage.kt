package com.example.geminichatbot.ui.theme

import ChatViewModel
import android.content.Context
import android.os.Build
import android.widget.Toast
import androidx.annotation.RequiresApi
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ExitToApp
import androidx.compose.material.icons.filled.MoreVert
import androidx.compose.material.icons.filled.Send
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.colorResource
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.Font
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.navigation.NavController
import com.example.geminichatbot.MessageModel
import com.example.geminichatbot.R
import com.google.android.gms.auth.api.signin.GoogleSignIn
import com.google.android.gms.auth.api.signin.GoogleSignInOptions
import com.google.firebase.Firebase
import com.google.firebase.auth.auth

val messageList by lazy {
    mutableStateListOf<MessageModel>()
}

@RequiresApi(value = Build.VERSION_CODES.VANILLA_ICE_CREAM)
@Composable
fun ChatPage(
    modifier: Modifier = Modifier,
    navController: NavController,
    authViewModel: AuthViewModel,
    viewModel: ChatViewModel
) {

    val context = LocalContext.current

    Scaffold(
        modifier = Modifier.fillMaxSize(),
        contentWindowInsets = WindowInsets(0),
        containerColor = colorResource(R.color.highDarkBlue)
    ) { paddingValues ->
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(paddingValues)
                    .imePadding(),

            ) {
                // Your existing functions
                AppHeader(navController)

                MessageList(
                    modifier = Modifier.weight(1f),
                    messageList = viewModel.messageList
                )

                MessageInput(
                    onMessageSend = { question ->
                        // 2. Pass the context to your viewModel function
                        viewModel.sendMessage(question = question, context = context)
                    }
                )
            }


    }
}

fun signOut (context: Context, webClientId: String, onComplete: () -> Unit){
    Firebase.auth.signOut()
    val googleSignInOptions = GoogleSignInOptions.Builder(GoogleSignInOptions.DEFAULT_SIGN_IN)
        .requestIdToken(webClientId)
        .requestEmail()
        .build()

    val googleSignInclient = GoogleSignIn.getClient(context, googleSignInOptions)
    googleSignInclient.signOut().addOnCompleteListener {
        onComplete()
    }
}

@Composable
fun MessageList(modifier: Modifier = Modifier, messageList: List<MessageModel>){

    LazyColumn(
        modifier = modifier,
        reverseLayout = true
    ) {
        items(messageList.reversed()){
            MessageRow(messageModel = it)
        }
    }

}

fun formatGeminiResponse(text: String): AnnotatedString {
    return buildAnnotatedString {
        // First, split the text by the bold delimiter "**"
        val boldParts = text.split("**")

        boldParts.forEachIndexed { index, boldPart ->
            if (index % 2 == 1) {
                // This part is bold
                withStyle(style = SpanStyle(fontWeight = FontWeight.Bold)) {
                    append(boldPart)
                }
            } else {
                // This part is not bold, so now check for italics within it
                val italicParts = boldPart.split("*")

                italicParts.forEachIndexed { italicIndex, italicPart ->
                    if (italicIndex % 2 == 1) {
                        // This sub-part is italic
                        withStyle(style = SpanStyle(fontStyle = FontStyle.Italic)) {
                            append(italicPart)
                        }
                    } else {
                        // This sub-part is normal text
                        append(italicPart)
                    }
                }
            }
        }
    }
}


@Composable
fun MessageRow(messageModel: MessageModel) {
    val isModel = messageModel.role == "model"

    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .padding(
                    start = if (isModel) 8.dp else 70.dp,
                    end = if (isModel) 70.dp else 8.dp
                )
        ) {
            Box(
                modifier = Modifier
                    .padding(top = 8.dp, bottom = 8.dp)
                    .align(
                        alignment = if (isModel) Alignment.BottomStart else Alignment.BottomEnd
                    )
                    .background(
                        color = if (isModel) colorResource(R.color.blue2) else Color.LightGray,
                        shape = RoundedCornerShape(20.dp)
                    )
                    .padding(12.dp),
            ) {
                Text(
                    formatGeminiResponse(messageModel.message),
                    fontFamily = FontFamily(Font(R.font.sfpromedium)),
                    color = if (isModel) Color.White else Color.Black
                )
            }
        }
    }
}

@Composable
fun MessageInput(onMessageSend : (String) -> Unit){

    var message by remember {
        mutableStateOf("")
    }
    Row(
        modifier = Modifier.padding(8.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        OutlinedTextField(
            modifier = Modifier.weight(1f),
            shape = RoundedCornerShape(25.dp),
            colors = OutlinedTextFieldDefaults.colors(
                unfocusedContainerColor = Color.Black,
                focusedContainerColor = Color.Black,
                focusedBorderColor = Color.DarkGray,
                unfocusedBorderColor = Color.Transparent,
                cursorColor = Color.Gray,
            ),
            value = message,
            placeholder = {
                Text(text = "Ask anything", fontFamily = FontFamily(Font(R.font.sfproregular)))
            },
            textStyle = TextStyle(fontFamily = FontFamily(Font(R.font.sfproregular)), fontSize = 18.sp),

            onValueChange = {message = it})
        IconButton(onClick = {
            if(message.isNotEmpty()){
                onMessageSend(message)
                message = ""
            }

        } ) {
            Icon(imageVector = Icons.Default.Send, contentDescription = "send", tint = Color.Gray)
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AppHeader(navController: NavController) {
    val context = LocalContext.current
    val webClientId = "495370860306-o3oiom8urecku6gt58ss0rfd2jov3vo1.apps.googleusercontent.com"
    var showMenu by remember { mutableStateOf(false) }

    Column {


        TopAppBar(
            title = {
                Text(text = "MedX",
                    fontFamily = FontFamily(Font(R.font.sfprobold)),
                    fontWeight = FontWeight.ExtraBold,
                    fontSize = 30.sp,
                    color = Color.White)
            },
            colors = TopAppBarDefaults.topAppBarColors(
                containerColor = colorResource(R.color.blue2),
                titleContentColor = Color.White,
                actionIconContentColor = Color.White
            ),
            actions = {
                IconButton(onClick = { showMenu = !showMenu }) {
                    Icon(
                        imageVector = Icons.Default.MoreVert,
                        contentDescription = "Menu",
                        tint = Color.White

                    )
                }
                DropdownMenu(
                    expanded = showMenu,
                    onDismissRequest = {showMenu = false}
                ) {
                    DropdownMenuItem(
                        text = {
                            Text("Sign Out", color = Color.White, fontWeight = FontWeight.Medium)
                        }, onClick = {
                            showMenu = false
                            signOut(context, webClientId, onComplete = {
                                navController.navigate(route = "login"){
                                    popUpTo(navController.graph.id) {
                                        inclusive = true
                                    }
                                    launchSingleTop = true
                                }
                                Toast.makeText(context, "Signed out successfully", Toast.LENGTH_SHORT).show()
                            })


                        },
                        leadingIcon = {
                            Icon(
                                imageVector = Icons.AutoMirrored.Filled.ExitToApp,
                                contentDescription = "Sign Out",
                                tint = Color.Red
                            )
                        }

                    )
                }
            }
        )

    }
}