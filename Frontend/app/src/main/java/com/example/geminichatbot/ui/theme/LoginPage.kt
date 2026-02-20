package com.example.geminichatbot.ui.theme

import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.livedata.observeAsState
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.Font
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.navigation.NavController
import com.example.geminichatbot.R
import com.google.android.gms.auth.api.signin.GoogleSignIn
import com.google.android.gms.auth.api.signin.GoogleSignInOptions
import com.google.android.gms.common.SignInButton
import com.google.firebase.Firebase
import com.google.firebase.auth.GoogleAuthProvider
import com.google.firebase.auth.auth

@Composable
fun LoginPage(modifier: Modifier = Modifier, navController: NavController, authViewModel: AuthViewModel){

    var email by remember {
        mutableStateOf("")
    }

    var password by remember {
        mutableStateOf("")
    }

    val webClientId = "495370860306-sk35u38gaqm2aaq77mmqdeqh4s5fv2au.apps.googleusercontent.com"

    val authState = authViewModel.authState.observeAsState()
    val context = LocalContext.current

    val gooogleSignInOptions = remember {
        GoogleSignInOptions.Builder(GoogleSignInOptions.DEFAULT_SIGN_IN)
            .requestIdToken(webClientId)
            .requestEmail()
            .build()
    }

    val googleSignInClient = remember {
        GoogleSignIn.getClient(context, gooogleSignInOptions)
    }

    val launcher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.StartActivityForResult()
    ) {result ->
        val task = GoogleSignIn.getSignedInAccountFromIntent(result.data)
        try {
            val account = task.result
            val credential = GoogleAuthProvider.getCredential(account.idToken, null)
            Firebase.auth.signInWithCredential(credential)
                .addOnCompleteListener { task ->
                    if(task.isSuccessful){
                        Toast.makeText(context, "Google sign in successful", Toast.LENGTH_SHORT).show()
                        navController.navigate("chat"){
                            popUpTo("login") {
                                inclusive = true
                            }
                        }
                    }else{
                        Toast.makeText(context, "Google sign in failed", Toast.LENGTH_SHORT).show()

                    }

                }
        }catch (e: Exception){
            Toast.makeText(context, "Google sign in failed: ${e.message}", Toast.LENGTH_SHORT).show()
        }

    }

    LaunchedEffect(authState.value) {
        when(authState.value){
            is AuthState.Authenticated -> navController.navigate("chat"){
                popUpTo("login") {
                    inclusive = true
                }
            }
            is AuthState.Error -> Toast.makeText(context, (authState.value as AuthState.Error).message, Toast.LENGTH_SHORT).show()
            else -> Unit
        }
    }

    Box(
        modifier = Modifier.fillMaxSize()
    ){
        Image(
            painter = painterResource(id = R.drawable.chat_page_bg),
            contentDescription = "chatbg",
            modifier = Modifier.fillMaxSize(),
            contentScale = ContentScale.Crop
        )

        Column(
            modifier = modifier.fillMaxSize(),
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(text = "Login", fontSize = 40.sp, fontFamily = FontFamily(Font(R.font.sfprobold)))
            Spacer(modifier = Modifier.height(16.dp))
            OutlinedTextField(
                colors = OutlinedTextFieldDefaults.colors(
                    unfocusedContainerColor = Color.Black,
                    focusedContainerColor = Color.Black,
                    focusedBorderColor = Color.DarkGray,
                    unfocusedBorderColor = Color.Transparent,
                    cursorColor = Color.Gray,
                ),
                shape = RoundedCornerShape(15.dp),
                value = email, onValueChange = {
                email = it
            }, label = {
                Text(text = "Email", fontFamily = FontFamily(Font(R.font.sfproregular)))
            }
            )
            Spacer(modifier = Modifier.height(8.dp))
            OutlinedTextField(
                colors = OutlinedTextFieldDefaults.colors(
                    unfocusedContainerColor = Color.Black,
                    focusedContainerColor = Color.Black,
                    focusedBorderColor = Color.DarkGray,
                    unfocusedBorderColor = Color.Transparent,
                    cursorColor = Color.Gray,
                ),
                shape = RoundedCornerShape(15.dp),
                value = password, onValueChange = {
                password = it
            }, label = {
                Text(text = "Password", fontFamily = FontFamily(Font(R.font.sfproregular)))
            }
            )
            Spacer(modifier = Modifier.height(16.dp))

            Button(onClick = {
                authViewModel.login(email, password)
            },
                colors = ButtonDefaults.buttonColors(
                    containerColor = Color.White,
                    contentColor = Color.Black
                )) {
                Text(text = "Login", fontFamily = FontFamily(Font(R.font.sfpromedium)))
            }

            Spacer(modifier = Modifier.height(20.dp))

            Text(text = "OR", fontSize = 15.sp, fontFamily = FontFamily(Font(R.font.sfpromedium)))
            Spacer(modifier = Modifier.height(16.dp))


            CustomGoogleSignInButton(
                onClick = {
                    val signInIntent = googleSignInClient.signInIntent
                    launcher.launch(input = signInIntent)
                }
            )

        }


    }



}
@Composable
fun CustomGoogleSignInButton(
    onClick: () -> Unit
) {
    Button(
        onClick = onClick,
        // Customize the size as you wish
        modifier = Modifier
            .width(250.dp)
            .height(50.dp),
        // Customize the colors
        colors = ButtonDefaults.buttonColors(
            containerColor = Color.White, // Your desired background color
            contentColor = Color.Black    // Your desired text/icon color
        ),
        // Add a border, elevation, etc. if you want
        elevation = ButtonDefaults.buttonElevation(defaultElevation = 8.dp)
    ) {
        Icon(
            painter = painterResource(id = R.drawable.google),
            contentDescription = "Google sign-in icon",
            modifier = Modifier.size(24.dp),
            // Important: Set tint to Color.Unspecified to use the icon's original colors
            tint = Color.Unspecified
        )
        Spacer(modifier = Modifier.width(16.dp))
        Text("Sign in with Google", fontFamily = FontFamily(Font(R.font.sfprobold)), fontSize = 15.sp)
    }
}