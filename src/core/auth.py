"""
Login simple por st.secrets para proteger el acceso en la nube.

Usuarios en st.secrets['usuarios'] como pares usuario = "contraseña".
En local (sin secrets de usuarios) NO se exige login: la app abre directo,
para no estorbar el flujo de escritorio.

Si más adelante se necesitan roles, se agregan aquí; por ahora basta con
proteger el acceso. Las claves van en los Secrets de Streamlit, NUNCA en el repo.
"""
from __future__ import annotations


def _usuarios_configurados() -> dict:
    try:
        import streamlit as st
        if "usuarios" in st.secrets:
            return dict(st.secrets["usuarios"])
    except Exception:
        pass
    return {}


def login_requerido() -> bool:
    """True si hay usuarios configurados (entonces se exige login)."""
    return len(_usuarios_configurados()) > 0


def gate() -> bool:
    """Muestra el formulario de login si hace falta. Devuelve True si el
    acceso está permitido (autenticado o sin login configurado).

    Llamar al inicio del Home. Si devuelve False, hacer st.stop().
    """
    import streamlit as st

    usuarios = _usuarios_configurados()
    if not usuarios:
        return True  # sin login configurado (local) → acceso libre

    if st.session_state.get("_auth_ok"):
        return True

    st.markdown("### 🔒 Acceso · SBC Demanda")
    st.caption("Ingresa tus credenciales para continuar.")
    with st.form("login_form"):
        usuario = st.text_input("Usuario")
        clave = st.text_input("Contraseña", type="password")
        ok = st.form_submit_button("Entrar", type="primary")
    if ok:
        if usuario in usuarios and str(usuarios[usuario]) == clave:
            st.session_state["_auth_ok"] = True
            st.session_state["_auth_user"] = usuario
            st.rerun()
        else:
            st.error("Usuario o contraseña incorrectos.")
    return False


def usuario_actual() -> str:
    import streamlit as st
    return st.session_state.get("_auth_user", "")


def logout_boton():
    """Botón de cerrar sesión (opcional, para el sidebar)."""
    import streamlit as st
    if st.session_state.get("_auth_ok"):
        if st.sidebar.button("🔓 Cerrar sesión"):
            st.session_state.pop("_auth_ok", None)
            st.session_state.pop("_auth_user", None)
            st.rerun()
