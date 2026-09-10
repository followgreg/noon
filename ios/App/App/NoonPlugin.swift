import UIKit
import Capacitor
import OneSignalFramework

@objc(NoonPlugin)
public class NoonPlugin: CAPPlugin, CAPBridgedPlugin {
    public let identifier = "NoonPlugin"
    public let jsName = "NoonPlugin"
    public let pluginMethods: [CAPPluginMethod] = [
        CAPPluginMethod(name: "checkAndPromptNotifications", returnType: CAPPluginReturnPromise)
    ]

    @objc func checkAndPromptNotifications(_ call: CAPPluginCall) {
        let promptKey = "noonHasShownPushPrompt"
        let resultKey = "noonHasSeenResult"

        guard !UserDefaults.standard.bool(forKey: promptKey),
              !UserDefaults.standard.bool(forKey: resultKey) else {
            call.resolve()
            return
        }

        UserDefaults.standard.set(true, forKey: promptKey)
        UserDefaults.standard.set(true, forKey: resultKey)

        DispatchQueue.main.async {
            guard let vc = self.bridge?.viewController else {
                call.resolve()
                return
            }
            let alert = UIAlertController(
                title: "Never miss a NOON",
                message: "Get notified every day at noon when your puzzle is ready.",
                preferredStyle: .alert
            )
            alert.addAction(UIAlertAction(title: "Turn On Notifications", style: .default) { _ in
                OneSignalManager.shared.requestPushPermission()
            })
            alert.addAction(UIAlertAction(title: "Not Now", style: .cancel, handler: nil))
            vc.present(alert, animated: true) {
                call.resolve()
            }
        }
    }
}
