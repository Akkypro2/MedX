package com.example.geminichatbot.ui.theme

// Core Compose imports
import android.os.Build
import androidx.annotation.RequiresApi
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier  // Fixed - Compose Modifier, not Java reflection

// ViewModel imports
import androidx.lifecycle.viewmodel.compose.viewModel

// Navigation imports
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController

@RequiresApi(Build.VERSION_CODES.VANILLA_ICE_CREAM)
@Composable
fun MyAppNavigation(modifier: Modifier = Modifier, authViewModel: AuthViewModel){
    val navController = rememberNavController()
    NavHost(navController = navController, startDestination = "login", builder = {
        composable("login") {
            LoginPage(modifier, navController, authViewModel)
        }
        composable("chat") {
            ChatPage(modifier, navController, authViewModel, viewModel())
        }
    })
}